from setuptools import setup

setup(
    name="SponsorDetect",
    version="0.2",
    description="Recognize sponsored content in youtube videos",
    url="https://github.com/yannikkellerde/Sponsor-Detect",
    author="Yannik Keller, Jan Mackensen and Jonas Stadtmüller",
    author_email="yannik.keller@stud.tu-darmstadt.de",
    license="MIT",
    packages=["bilstm","frontend"],
    zip_safe=False,
)