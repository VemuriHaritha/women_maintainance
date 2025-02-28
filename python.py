import google.generativeai as genai
genai.configure(api_key="AIzaSyBurXb_5_DHKQRVSjHtVQeSdyhiNlDCRGA")

for model in genai.list_models():
    print(model.name)
