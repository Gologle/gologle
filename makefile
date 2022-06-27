clean:
	rm -f *.db

test:
	python -m pytest

run:
	uvicorn src.main:app --reload

BENCH_MEM_PATH := benchmarks/memory

bench-mem-vect: clean
# Vectorial model memory profiling
# Receives as argument 'd' the target dataset
# Results are stored in a HTML file
# Example:
# 	make bench-mem-vect d=cranfield
# 	make bench-mem-vect d=newsgroups
	python -m memray run -f \
		-o ${BENCH_MEM_PATH}/vectorial_$(d).bin \
		${BENCH_MEM_PATH}/mem_vectorial_$(d).py
	python -m memray flamegraph -f \
		${BENCH_MEM_PATH}/vectorial_$(d).bin
	firefox ${BENCH_MEM_PATH}/memray-flamegraph-vectorial_$(d).html

bench-mem:
	bench-mem-vect d=cranfield
	bench-mem-vect d=newsgroups
