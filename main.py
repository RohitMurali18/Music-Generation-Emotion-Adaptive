from emotion_analysis import inference,data_preprocessing,EATS

prompt = "i am walking down a road and i see a rainbow and it is sunny. i love life. then suddenly it turns dark and cloudy. it starts raining and i start crying."
# predictions = inference.predict_all_labels(prompt)
# print(predictions)
segmentation = data_preprocessing.segment_text(prompt)
n = len(segmentation)
allpredictions = []
for i in range(0,n):
    predictions = inference.predict(segmentation[i])
    allpredictions.append(predictions)

mapping = EATS.get_music_params(allpredictions)
print(mapping)
