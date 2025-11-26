from openai import OpenAI
client = OpenAI()

file_response = client.files.content("file-EyThzm9Betdv7K4ppbHAxH")
print(file_response.text)