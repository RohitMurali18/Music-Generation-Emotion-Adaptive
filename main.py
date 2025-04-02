from emotion_analysis import inference
prompt = "idk why i started yapping then my bf got really mad at me and wanted to break up with me. what do I even do now. I'm such a mess, gonna go get some wine and then drunk call all my friends"
predictions = inference.predict(prompt)
print(predictions)
