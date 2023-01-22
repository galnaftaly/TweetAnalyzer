import re
def regex_a(text):
    m = re.search(r'Review:\s\d+', text)
    return m

def regex_b(text):
    m = re.search(r'Review:\s[1-5]', text)
    return m

def regex_c(text):
    m = re.search(r'Review:\s*[1-5]', text)
    return m

def regex_d(text):
    m = re.search(r'[^\n]{3,5}\.', text)
    return m

text = "Movie Review:10, Movie Review: 20"
text2 = "hello my name. is gal"
print(regex_a(text))
print(regex_b(text))
print(regex_c(text))
print(regex_d(text2))