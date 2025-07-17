

import requests
import io
import numpy as np
from PIL import Image, ImageStat

url="http://legion-pro-7-16arx8h:8080/v1/data-persistance/getimage?isic_id=7748442"

response = requests.get(url)

img_stream = io.BytesIO(response.content)

img=Image.open(img_stream)

img_np=np.array(img)

print(img_np.shape)
bins=np.arange(0,256,25)
bins[-1]=255
print( np.histogram(img_np[...,0], bins)[0]/(img_np.size//3) )
print( np.histogram(img_np[...,1], bins)[0]/(img_np.size//3) )
print( np.histogram(img_np[...,2], bins)[0]/(img_np.size//3) )

histogram_data = img.histogram()

#print(len(histogram_data))

#print(histogram_data)

stat = ImageStat.Stat(img)

print(str(stat))

print(f"Mean: {stat.mean}")
print(f"Median: {stat.median}")
print(f"Extrema (min, max): {stat.extrema}")
print(f"Standard Deviation: {stat.stddev}")
print(f"Variance: {stat.var}")

#img.save('img.png')

#from IPython import embed as  idbg;  idbg(colors='Linux') 

