from distutils.core import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os

def install_gpu():
    try:
        subprocess.check_output("nvidia-smi")
        os.environ["CFLAGS"] = "-O3 -gstabs -march=native"
        os.environ["LDFLAGS"] = "-O3 -gstabs -march=native"
        module1 = Extension(
            "SimulatorGPU",
            sources=["wrapper.cu"],
            extra_compile_args=[
                "-O3",
                "--std",
                "c++17",
                "-arch=compute_86",
                "-code=sm_86",
                "-Xptxas=-O3",
                "--compiler-options",
                "-fPIC",
                "--compiler-options",
                "-march=native",
                "--compiler-options",
                "-finline-functions",
                "--compiler-options",
                "-std=c++17",
            ],
        )

        class CUDA_build_ext(build_ext):
            """
            Custom build_ext command that compiles CUDA files.
            Note that all extension source files will be processed with this compiler.
            """

            def build_extensions(self):
                self.compiler.src_extensions.append(".cu")
                self.compiler.set_executable("compiler_so", "nvcc")
                self.compiler.set_executable("linker_so", "nvcc --shared")
                if hasattr(self.compiler, "_c_extensions"):
                    self.compiler._c_extensions.append(".cu")  # needed for Windows
                self.compiler.spawn = self.spawn
                build_ext.build_extensions(self)

        print("SimulatorGPU is being built!")
        setup(
            name="SimulatorGPU",
            version="1.0",
            description="GPU version of the ARC simulator",
            ext_modules=[module1],
            cmdclass={"build_ext": CUDA_build_ext},
        )
    except Exception:
        print("No GPU version available")

def install_cpu():
    print("SimulatorCPU is being built!")
    os.environ["CFLAGS"] = "-O3 -march=native"
    os.environ["LDFLAGS"] = "-O3 -march=native"

    moduleCPU = Extension(
            "SimulatorCPU",
            sources=["wrapper.cpp"],
            extra_compile_args=["-g", "-O3", "-march=native", "-std=c++17", "-DSETUP_BUILDING_CPU", "-Wsign-compare", "-finline-functions"],
    )
    setup(
            name="SimulatorCPU",
            version="1.0",
            description="Fast version of the ARC simulator",
            ext_modules=[moduleCPU],
    )

if __name__ == "__main__":
    install_gpu()
    #install_cpu()
