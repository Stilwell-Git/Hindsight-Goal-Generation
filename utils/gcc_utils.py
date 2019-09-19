import os
import ctypes
import numpy as np

def c_double(value):
	return ctypes.c_double(value)
def c_int(value):
	return ctypes.c_int(value)

def gcc_complie(c_path, so_path=None):
	assert c_path[-2:]=='.c'
	if so_path is None:
		so_path = c_path[:-2]+'.so'
	else:
		assert so_path[-3:]=='.so'
	os.system('gcc -o '+so_path+' -shared -fPIC '+c_path+' -O2')
	return so_path

def gcc_load_lib(lib_path):
	if lib_path[-2:]=='.c':
		lib_path = gcc_complie(lib_path)
	else:
		assert so_path[-3:]=='.so'
	return ctypes.cdll.LoadLibrary(lib_path)