default:
	cd source; sage --pkg ./asc-0.1
	mv source/asc-0.1.spkg ./build

install: default
	cd build; sage -f ./asc-0.1.spkg

clean:
	rm ./build/*
