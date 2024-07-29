from distutils.core import setup, Extension
from setuptools.command.build_ext import build_ext
import os

os.environ['CFLAGS'] = '-O3 -gstabs -march=native'
os.environ['LDFLAGS'] = '-O3 -gstabs -march=native'

module1 = Extension('SimulatorGPU',
                    sources = ['wrapper.cu'],
                    extra_compile_args = ["-O3", "--compiler-options", "-fPIC"])

class CUDA_build_ext(build_ext):
    """
    Custom build_ext command that compiles CUDA files.
    Note that all extension source files will be processed with this compiler.
    """
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        self.compiler.set_executable('compiler_so', 'nvcc')
        self.compiler.set_executable('linker_so', 'nvcc --shared')
        if hasattr(self.compiler, '_c_extensions'):
            self.compiler._c_extensions.append('.cu')  # needed for Windows
        self.compiler.spawn = self.spawn
        build_ext.build_extensions(self)


setup (name = 'SimulatorGPU',
       version = '1.0',
       description = 'GPU version of the ARC simulator',
       ext_modules = [module1],
      cmdclass={'build_ext': CUDA_build_ext})