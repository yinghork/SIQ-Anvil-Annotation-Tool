from PIL import Image, ImageDraw, ImageFont

import numpy as np
import json

def clean_attention(data):
    
#     with open(infile, 'r') as f:
#         data = json.load(f)

    #################### Combine 's ################################################
    new_data = {"word": {}, "attention": {}}
    prev_word = None
    prev_att = None
    count = 0
    for i, word in data["word"].items():
        if word.startswith("'"):
            if prev_word is not None:
                combined_word = prev_word + word
                average_att = (prev_att + data["attention"][i]) / 2
                new_data["word"][str(count-1)] = combined_word
                new_data["attention"][str(count-1)] = average_att
            prev_word = None
            prev_att = None
        else:
            new_data["word"][str(count)] = word
            new_data["attention"][str(count)] = data["attention"][i]
            prev_word = word
            prev_att = data["attention"][i]
            count += 1
    data = new_data

    #################### Combine split words #######################################
    new_word = {}
    new_attention = {}
    ignore_words = ['<pad>', '<s>', '</s>', ',', '!', '.', '?']
    i = 0
    while i < len(data["word"]):
        word = data["word"][str(i)]
        attention = data["attention"][str(i)]
        if word[0] != "\u0120" and word not in ignore_words:
            j = i + 1
            while j < len(data["word"]) and (data["word"][str(j)][0] != "\u0120"
                        and data["word"][str(j)][0] not in ignore_words):
                word += data["word"][str(j)]
                attention = (attention + data["attention"][str(j)]) / 2.0
                j += 1
            i = j
        else:
            i += 1
        new_word[str(len(new_word))] = word
        new_attention[str(len(new_attention))] = attention
    data["word"] = new_word
    data["attention"] = new_attention

    #################### Remove special characters #################################
    new_word_dict = {}
    new_attention_dict = {}
    ignore_words = ['<pad>', ',', '!', '.', '?', 'Ġ']
    for key in data["word"]:
        if data["word"][key] not in ignore_words:
            new_word_dict[key] = data["word"][key]
            new_attention_dict[key] = data["attention"][key]
    for key in new_word_dict:
        new_word_dict[key] = new_word_dict[key].replace("Ġ", "")
    new_data = {"word": new_word_dict, "attention": new_attention_dict}
    data = new_data

    #################### Correct the keys ##########################################
    new_word = {}
    j = 0
    for i, word in enumerate(data["word"].values()):
        new_word[str(j)] = word
        j += 1
    data["word"] = new_word
    new_attention = {}
    j = 0
    for i, value in enumerate(data["attention"].values()):
        new_attention[str(j)] = value
        j += 1
    data["attention"] = new_attention

    #################### Format data ###############################################
    new_data = []
    j = 0
    while j != len(data['word']):
        #question
        curr = 1
        q = []
        q_vals = []
        while data['word'][str(j+curr)] != "</s>":
            q.append(data['word'][str(j+curr)])
            q_vals.append(data['attention'][str(j+curr)])
            curr += 1
        j = j + curr + 1
        #correct
        curr = 1
        a = []
        a_vals = []
        while data['word'][str(j+curr)] != "</s>":
            a.append(data['word'][str(j+curr)])
            a_vals.append(data['attention'][str(j+curr)])
            curr += 1
        j = j + curr + 1
        #question repeat
        curr = 1
        while data['word'][str(j+curr)] != "</s>":
            q_vals[curr-1] = (q_vals[curr-1] + data['attention'][str(j+curr)]) / 2
            curr += 1
        j = j + curr + 1
        #incorrect
        curr = 1
        i = []
        i_vals = []
        while data['word'][str(j+curr)] != "</s>":
            i.append(data['word'][str(j+curr)])
            i_vals.append(data['attention'][str(j+curr)])
            curr += 1
        j = j + curr + 1
        #append
        new_data.append({'q': q, 'q_vals': q_vals, 
                         'a': a, 'a_vals': a_vals, 
                         'i': i, 'i_vals': i_vals})    

    #################### New json file #############################################
#     with open(outfile, "w") as f:
#         json.dump(new_data, f)

    return new_data


# Given array of words and array of attention values (should be same length),
# create gradient image according to corresponding attention values
def text2img(texts, attentions):
    diff = max(attentions) - min(attentions)
    ranges = [160, 120, 80, 40, 0, 0]
    grads = [ranges[int((attentions[i] - min(attentions)) * 5 / diff)] for i in range(len(attentions))]
    total_len = 6 + 6 * len(texts)
    for i in range(len(texts)): total_len += len(texts[i]) * 6
    img = Image.new('RGB', (total_len, 15), (255, 255, 255))
    d = ImageDraw.Draw(img)
    index = 3
    for i in range(len(texts)):
        text, grad = texts[i], grads[i]
        nextindex = index + len(text) * 6 
        d.rectangle([(index-3, 0), (nextindex+3, 15)], fill=(grad, grad, 255))
        d.text((index, 0), text, fill=(0, 0, 0))
        index = nextindex + 6
    return img

# Given question, correct, and incorrect answer word arrays and their attention arrays,
# output 3 gradient images
def generateAttentionImgs(q, a, i, qval, aval, ival):
    qimg = text2img(q, qval)
    aimg = text2img(a, aval)
    iimg = text2img(i, ival)
    return qimg, aimg, iimg