import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="data_fast_insights",
    version="0.2",
    author="p.gafiatullin; a.tolmachev",
    author_email="p.gafiatullin@xsolla.com",
    description="Library for using Data Fast Insights model on custom data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xsolla",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
