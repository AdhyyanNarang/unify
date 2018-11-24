import urllib.request

#f_dog = open('dog_url.txt', 'r')
#dog_urls = f_dog.readlines()

f_fish = open('fish_url.txt', 'r')
fish_urls = f_fish.readlines()

"""
print('Starting to save dog images')
bad_dog_counter = 0

for (i, url) in enumerate(dog_urls):
    try:
        urllib.request.urlretrieve(url, './Data/rawImages/main_dog/dog_' + str(i) + '.jpg')
    except:
        bad_dog_counter += 1
        print('bad image number: ' + str(bad_dog_counter))
"""

print('Starting to save fish images')
bad_fish_counter = 0

for (i, url) in enumerate(fish_urls):
    try:
        urllib.request.urlretrieve(url, './Data/rawImages/fish/fish_' + str(i) + '.jpg')
    except:
        bad_fish_counter += 1
        print('bad image number: ' + str(bad_fish_counter))

f = open('summary.txt', 'w')
f.write('Number of bad dog images:' + str(bad_dog_counter))
f.write('Number of bad fish images:' + str(bad_fish_counter))
