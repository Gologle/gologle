from src.parsers import NewsgroupsParser
from src.engines.vectorial import VectorialModel

ngp = NewsgroupsParser()
m = VectorialModel(ngp)
