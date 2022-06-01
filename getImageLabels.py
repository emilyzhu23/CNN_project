import requests 
import shutil
from bs4 import BeautifulSoup 
    
def getdata(url): 
    r = requests.get(url) 
    return r.text 
    
htmldata = getdata("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html") 
soup = BeautifulSoup(htmldata, 'html.parser') 
for item in soup.find_all('img'):
    image_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/" + item['src']
    filename = image_url.split("/")[-1]
    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream = True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        
        # Open a local file with wb ( write binary ) permission.
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)
            
        print('Image sucessfully Downloaded: ',filename)
    else:
        print('Image Couldn\'t be retreived')