#!/bin/bash

if [ ${TRAVIS_OS_NAME} = "osx" ]
then
    ext="dylib"
elif [ ${TRAVIS_OS_NAME} = "linux" ]
then
    ext="so"
elif [ ${TRAVIS_OS_NAME} = "windows" ]
then
    ext="dll"
else
    echo "Warning: operating system not supported."
fi

mv libfasttransforms.${ext} libfasttransforms.${TRAVIS_TAG}.${CC}.${ext}
