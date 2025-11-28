from dataclasses import dataclass, asdict #딕션화

import datetime, time
from time import strftime, localtime, gmtime  #시간 변환 / gmtime으로 시간 통일
from typing import Any, List, Optional, Tuple # 타입 힌트
from langchain_core.documents import Document # 문서 객체
import numpy as np 

from remembr.memory.memory import Memory, MemoryItem # 기본 메모리 클래스와 아이템

from langchain_community.vectorstores import Milvus # vertor DB
from langchain_huggingface import HuggingFaceEmbeddings # 임베딩 모델

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


FIXED_SUBTRACT=1721761000 # this is just a large value that brings us close to 1970



class MilvusWrapper:

    def __init__(self, collection_name='test', ip_address='127.0.0.1', port=19530, drop_collection=False):
        self.collection_name = collection_name
        self.collection = self.connect_to_milvus_collection(collection_name, 1024, address=ip_address, port=port, drop_collection=drop_collection)


    def drop_collection(self):
        utility.drop_collection(self.collection_name)

    def connect_to_milvus_collection(self, collection_name, dim, address='127.0.0.1', port=19530, drop_collection=False):
        connections.connect(host=address, port=port)
        
        if drop_collection:
            utility.drop_collection(collection_name)
        
        fields = [
            FieldSchema(name='id', dtype=DataType.VARCHAR, description='ids', is_primary=True, auto_id=False, max_length=1000), # string/ids/primary/You provide unique IDs yourself when inserting or importing data/길이정의
            FieldSchema(name='text_embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim), # embedding model의 차원에 따라 상의 
            FieldSchema(name='position', dtype=DataType.FLOAT_VECTOR, description='position of robot', dim=3), # stores 32-bit floating-point numbers
            FieldSchema(name='theta', dtype=DataType.FLOAT, description='rotation of robot', dim=1), 
            FieldSchema(name='time', dtype=DataType.FLOAT_VECTOR, description='time', dim=2),
            FieldSchema(name='caption', dtype=DataType.VARCHAR, description='caption string', max_length=3000),

        ]
        schema = CollectionSchema(fields=fields, description='text image search') # 스키마 정의
        collection = Collection(name=collection_name, schema=schema) # 위에서 정의된 스키마를 바탕으로 컬렉션 생성

        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":1024} # 여기는 왜 또 1024지? 너무 큰것 같은데, Long 예제에서도 entity 갯수가 고작 330인데...
        }
        collection.create_index(field_name="text_embedding", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2} # 왜 2로만 했을까...이러면 index하는 효과가 없을 것 같은데..
        }
        collection.create_index(field_name="position", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="time", index_params=index_params)

        return collection
    
    def insert(self, data_list):
        res = self.collection.insert(data_list)

    def search(self, data):

        self.collection.load()

        BATCH_SIZE = 2 # 의미없어보임
        LIMIT = 10

        param = {
            "metric_type": "L2",
            "params": {
                "nprobe": 1024,
            }
        }

        res = self.collection.search(
            data=[data],
            anns_field="text_embedding",
            param=param,
            batch_size=BATCH_SIZE, # 의미없어보임
            limit=LIMIT, # 총 10개의 결과를 반환
            # expr="id > 3", # 필터링 조건 검색 조건들 중에서 id가 3보다 큰 것들만 고려하겠다.
            output_fields=["id", "text_embedding"]
        )

        return res




class MilvusMemory(Memory):


    def __init__(self, db_collection_name: str, db_ip='127.0.0.1', db_port=19530, time_offset=FIXED_SUBTRACT):

        self.db_collection_name = db_collection_name
        self.db_ip = db_ip
        self.db_port = db_port
        self.time_offset = time_offset

        self.embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')

        self.working_memory = []

        self.reset(drop_collection=False)


    def insert(self, item: MemoryItem, text_embedding=None):

        memory_dict = asdict(item) # dataclass -> dict (caption, time, position, theta)
        memory_dict['id'] = str(time.time()) # 현재 컴퓨터의 time 기반으로 고유 id 생성

        if text_embedding is None:
            text_embedding = self.embedder.embed_query(memory_dict['caption']) #dict(caption=...) -> embedding vector

        memory_dict['time'] =  [(memory_dict['time'] - self.time_offset), 0.0] # Time vecotrization/ dict(time=...) -> [time-time_offset, 0.0]

        memory_dict['text_embedding'] = text_embedding 

        self.milv_wrapper.insert([memory_dict]) # 삽입완료.

    def get_working_memory(self) -> list[MemoryItem]:
        return self.working_memory

    def reset(self, drop_collection=True):

        if drop_collection:
            print("Resetting memory. We are dropping the current collection")

        self.milv_wrapper = MilvusWrapper(self.db_collection_name, self.db_ip, self.db_port, drop_collection=drop_collection)

        text_vector_db = Milvus(
            self.embedder,
            connection_args={"host": self.db_ip, "port": self.db_port}, # 연결주소
            collection_name=self.db_collection_name, # 컬렉션 이름
            vector_field='text_embedding', 
            text_field='caption',
        )
        self.text_retriever = text_vector_db.as_retriever(search_kwargs={"k": 5}) #상위 5개 검색


        self.position_vector_db = Milvus(
            self.embedder, # we will ignore this
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='position',
            text_field='caption',
        )

        self.time_vector_db = Milvus(
            self.embedder, # we will ignore this
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='time',
            text_field='caption',
        )


    def search_by_position(self, query: tuple) -> str:
        # docs = pos_db.similarity_search_by_vector(np.array(query))
        docs = similarity_search_with_score_by_vector(self.position_vector_db, np.array(query).astype(float))

        self.working_memory += docs 

        docs = self.memory_to_string(docs)

        """Look up things online."""
        return docs

    def search_by_time(self, hms_time: str) -> str:

        # Input is time like 08:20:30
        # need to convert to searchable time
        t = gmtime(self.time_offset)
        mdy_date = strftime('%m/%d/%Y', t)
        template = "%m/%d/%Y %H:%M:%S"

        # if the hms_time is already in the mdy hms format without me doing anything, let's just use that.
        # bad llms don't listen :(
        try:
            res = bool(datetime.datetime.strptime(hms_time, template))
        except ValueError:
            res = False

        hms_time = hms_time.strip()
        if not res: # convert to the right format then
            hms_time = mdy_date + ' ' + hms_time

        query = time.mktime(datetime.datetime.strptime(hms_time,template).timetuple()) - self.time_offset
        # convert from hms_time to something searchable


        docs = similarity_search_with_score_by_vector(self.time_vector_db, np.array([query, 0]))

        self.working_memory += docs

        docs = self.memory_to_string(docs)
        # np.unique([doc.metadata['time'][0] for doc in docs])
        """Look up things online."""
        return docs



    def search_by_text(self, query: str) -> str: # text는 그냥 langchain으로 편하게 구할 수 있음.

        docs = self.text_retriever.invoke(query) # 상위 5개 검색
        
        self.working_memory += docs

        docs = self.memory_to_string(docs)

        """Look up things online."""
        return docs
    

    ### Doc formatting for the last LLM
    def memory_to_string(self, memory_list: list[MemoryItem], ref_time: float=None):
        if ref_time == None:
            ref_time = self.time_offset

        out_string = ""
        for doc in memory_list:
            if len(doc.metadata['time']) == 2:
                t = doc.metadata['time'][0]
            else:
                t = doc.metadata['time']
            
            if ref_time:
                t += ref_time
            t = gmtime(t)
            t = strftime('%Y-%m-%d %H:%M:%S', t)

            s = f"At time={t}, the robot was at an average position of {np.array(doc.metadata['position']).round(3).tolist()}."
            s += f"The robot average heading was {doc.metadata['theta']} in radian."
            s += f"The robot saw the following: {doc.page_content}\n\n"
            out_string += s
        return out_string


# NOTE: This version of the code returns the vector
def similarity_search_with_score_by_vector(
        pos_db,
        embedding: List[float],
        k: int = 5, # Defaults to 4.
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.6/Collection/search().md

        Args:
            embedding (List[float]): The embedding vector being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Tuple[Document, float]]: Result doc and score.
        """
        if pos_db.col is None:
            print("No existing collection to search.")
            return []

        if param is None:
            param = pos_db.search_params # default {'metric_type': 'L2', 'params': {'nprobe': 10}}

        # Determine result metadata fields with PK.
        output_fields = pos_db.fields[:] # ['id', 'text_embedding', 'position', 'theta', 'time', 'caption']
        # output_fields.remove(pos_db._vector_field) # NOTE: Only thing removed
        timeout = pos_db.timeout or timeout # default None
        # Perform the search.
        res = pos_db.col.search(
            data=[embedding],
            anns_field=pos_db._vector_field,
            param=param,
            limit=k,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        # Organize results.
        ret = []
        for result in res[0]: # res[0] corresponds to the first query vector 질문을 하나만 했으니까.
            data = {x: result.entity.get(x) for x in output_fields} # x in output_fields: 'id', 'text_embedding', 'position', 'theta', 'time', 'caption' 의 값들을 받아온다
            doc = pos_db._parse_document(data) # data dict -> Document 객체로 변환, text_field='caption' 이므로 page_content는 caption 값이 됨, 나머지 field들은 metadata로 들어감
            pair = (doc, result.score) # (Document, score) 튜플 생성
            ret.append(pair)

        return [doc for doc, _ in ret] # score는 무시하고 Document 객체들만 반환


