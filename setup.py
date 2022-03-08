import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='cdhtools',
    version='0.9.0',
    author='Stijn Kas',
    author_email='stijn.kas@pega.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/pegasystems/cdh-datascientist-tools',
    project_urls = {
        "Bug Tracker": "https://github.com/pegasystems/cdh-datascientist-tools/issues",
        "Wiki": "https://github.com/pegasystems/cdh-datascientist-tools/wiki"
    },
    license='Apache-2.0',
    packages=['cdhtools'],
    package_dir={'':'python'},
    install_requires=['pandas', 'plotly', 'seaborn', 'sklearn', 'requests', 'tqdm', 'nbformat'],
    keywords=['pega', 'pegasystems', 'cdh', 'cdhtools', 'customer decision hub', 'datascientist', 'tools'],
    python_required=">=3.6",
    classifiers=[
        'Development Status :: 4 - Beta'
    ]
)