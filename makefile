ifeq ($(OS),Windows_NT)
cc=gcc
else
cc=/opt/homebrew/bin/gcc-15
endif
prom=a

deps= $(wildcard *.h) #注意这样写的坏处是修改头文件的编译代价将非常大，但方便之处在于我们不必强迫自己写出互相独立的库文件
src = $(wildcard *.c)
obj= $(src:%.c=%.o)

INC=$(shell gsl-config --cflags)
LIB=

LIBS=$(shell gsl-config --libs)

ifeq ($(OS),Windows_NT)

endif

ifeq ($(OS),Windows_NT)
$(prom): $(obj)
	$(cc) -fopenmp -o $(prom) $(obj) $(LIB) -lgsl -lgslcblas
else
$(prom): $(obj)
	$(cc) -fopenmp $(obj) $(LIB) -o $(prom) $(LIBS) -lm
endif

%.o: %.c $(deps)
	$(cc) -fopenmp -c $(INC) $< -o $@

clean:
ifeq ($(OS),Windows_NT)
	del -rf $(obj) $(prom).exe
else
	rm -rf $(obj) $(prom)
endif