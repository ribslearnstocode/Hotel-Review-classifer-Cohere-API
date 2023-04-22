import cohere 
co = cohere.Client('gfIdClymE2F9VX7gJzNi83Rbt725D7DTpX29MFtf') 

reviews = ["I had a nice stay here in the hotel, the room was comfy, the pool was huge and the hotel bar was fantasic. Super good location right in the center of Los Angeles", "Great service, but I could not turn off the AC, the food was not as advertised though"]

for review in reviews:

  response = co.generate( 
    model='xlarge', 
    prompt='Review:\nThe view from the rooftop is beautiful and the lounge area is comfortable. The staff is polite and the location is convenient. However, the hotel is outdated and the shower could be cleaner. Additionally, the air conditioning cannot be adjusted and is always running.\nExtracted sentiment:\n{“Cleaning”: “Negative”, “AC”: “Negative”, “Room Quality”: “Negative”, “Service”: “Positive”, “View”: “Positive”, “Hotel Facilities”: “Positive”}\n--\nReview:\nThe rooms were cramped and barely passable as clean, but they were at least bearable. The front desk was swamped with guests and the wait to check-in was unbearable, but somehow the staff managed to maintain a façade of professionalism and friendliness. We would not recommend staying there again. Pool was big.\nExtracted sentiment:\n{“Rooms”: “Negative”, “Staff”: “Positive”, “Pool”: “Positive”}\n--\nReview:\nThis hotel is really clean and spacious, and the beds are comfortable. The breakfast was excellent, and the staff were friendly and helpful. The only downside is that the hotel is located far away from the attractions, and you can only use the hotel facilities after paying an additional fee.\nExtracted sentiment:\n{“Staff”: “Positive”, “Breakfast”: “Positive”, “Location”: “Negative”, “Hotel Facilities”: “Negative”, “Cleanliness”: “Positive”, “Value for money”: “Negative”}\n--\nReview:\n' + review + '\nExtracted sentiment:\n', 
    max_tokens=500, 
    temperature=0.7, 
    k=0, 
    p=0.75, 
    frequency_penalty=0, 
    presence_penalty=0, 
    stop_sequences=["--"], 
    return_likelihoods='NONE') 
  print('Prediction: {}'.format(response.generations[0].text)) 