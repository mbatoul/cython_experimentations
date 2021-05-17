import numpy
from Cython.Build import build_ext
from setuptools import Extension, setup

extensions = [
    Extension(
        "src/pairwise_dist/pairwise_dist",
        sources=["src/pairwise_dist/_pairwise_dist.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[
            "-fopenmp",
            "-O3",
            "-ftree-vectorize",
        ],
        extra_link_args=["-fopenmp"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="experimentations",
    cmdclass={"build_ext": build_ext},
    version="0.1",
    ext_modules=extensions,
    install_requires=["setuptools>=18.0", "cython>=0.27.3", "numpy"],
    python_requires=">=3.6",
)
