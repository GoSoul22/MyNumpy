from distutils.core import setup, Extension
import sysconfig

def main():
	CFLAGS = ['-g', '-Wall', '-std=c99', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
	LDFLAGS = ['-fopenmp']
	# Use the setup function we imported and set up the modules.
	# You may find this reference helpful: https://docs.python.org/3.6/extending/building.html
	patricstar_module = Extension('numc',
									extra_compile_args = CFLAGS,
									extra_link_args = LDFLAGS,
									library_dirs = [sysconfig.get_config_var("LIBDIR")],
									sources = ['numc.c', 'matrix.c'])
	setup(name = 'PatricStarPy',
		  version = '1.0',
		  description = 'PatricStar! PatricStar!! PatricStar!!!',
		  author = 'Justin Xia and Rixiao Zhang',
		  author_email = ' justinerxiabucks0119@berkeley.edu && rixiaozhang@berkeley.edu',
		  ext_modules = [patricstar_module])



if __name__ == "__main__":
    main()
