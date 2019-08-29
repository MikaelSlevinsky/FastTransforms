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
    echo "Error: operating system not supported."
    exit 1
fi

echo "Moving libfasttransforms.${ext} to libfasttransforms.${TRAVIS_TAG}.${CC}.${ext}"
mv libfasttransforms.${ext} libfasttransforms.${TRAVIS_TAG}.${CC}.${ext}
exit 0
