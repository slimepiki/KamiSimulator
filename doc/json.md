# Json System

This page is a manual for ```src/utilities/JsonParser``` which is a wrapper of [nlohmann json](https://github.com/nlohmann/json).

```JsonParser``` has get/set interfaces named ```SetVal(string key, T val, bool isSilent = false)```, and ```Get*```. Here, ```T``` is a template argument, and ```*``` is the corresponding type name. You should check the key exists before getting something by using ```CheckKey```.

This wrapper can manage the types listed below.

- ```bool```
- ```int```
- ```float```
- ```string```
- ```vector<bool>```
- ```vector<int>```
- ```vector<float>```
- ```vector<string>```.

That is, ```T``` should be one of them.

You can initialize with a .json file by using ```OpenJsonFile (string jsonPath)``` after class initialization or by initializing with  ```JsonParser(string jsonPath)``` directly.

```dump()``` provides JSON as string and ```bool SaveJson(string filePath, bool isSilent = false)``` saves the data as a .json file.
