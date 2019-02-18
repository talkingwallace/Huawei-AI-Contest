from google_images_download import google_images_download
import joblib
from joblib import Parallel, delayed

# ['糖炒栗子', 'layer cake', 'cheese', ]
# ['green bean cake', 'hot dog', 'french fries',]
# ['boiled corn', 'sampan porridge', 'poon choi', ]
# keywords = ['steak', '糯米糍', '黑芝麻糊',  ]
# ['pop corn', 'stir fried lettuce', 'Snowflake Flaky Pastry', 'frozen lemon tea', ]
#  ['黑米糕', 'braised pork', 'Omelette',]
# [ 'mango pomelo sago', '糍粑', 'onion rings']
# keywords = ['番茄炒蛋','煲仔饭','persimmon pancake',]

chromeDriverPath = r'C:\\Users\\talki\\Downloads\\chromedriver.exe'

def runAImageSpider(keyWords):
    response = google_images_download.googleimagesdownload()
    path = response.download({
        'keywords':keyWords,
        'limit':600,
        'chromedriver':chromeDriverPath
    })


# for i in keywords:
#     runAImageSpider(i)

runAImageSpider('冻柠茶')