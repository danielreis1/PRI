from urllib.request import urlopen
import feedparser
import xml.etree.ElementTree as ET

sites =	["http://www.nytimes.com/services/xml/rss/nyt/World.xml",\
		"http://www.nytimes.com/services/xml/rss/nyt/Africa.xml", "http://www.nytimes.com/services/xml/rss/nyt/Americas.xml",\
		"http://www.nytimes.com/services/xml/rss/nyt/AsiaPacific.xml", "http://www.nytimes.com/services/xml/rss/nyt/Europe.xml",\
		"http://www.nytimes.com/services/xml/rss/nyt/MiddleEast.xml", "http://rss.cnn.com/rss/edition_world.rss",\
		"http://feeds.washingtonpost.com/rss/world", "http://www.latimes.com/world/rss2.0.xml", "http://www.latimes.com/world/afghanistan-pakistan/rss2.0.xml",\
		"http://www.latimes.com/world/africa/rss2.0.xml", "http://www.latimes.com/world/mexico-americas/rss2.0.xml",\
		"http://www.latimes.com/world/asia/rss2.0.xml", "http://www.latimes.com/world/europe/rss2.0.xml", "http://www.latimes.com/world/middleeast/rss2.0.xml",\
		"http://www.latimes.com/world/worldnow/rss2.0.xml"]
		#Apaguei 'blog atwar' do NYT porque dava erro (moved permanently)

d = {}

for site in sites:
	print('\n\t***\t' + site + '\t***\t')
	d[site] = feedparser.parse(site)
	entries_keys = d[site]['entries'][0].keys()	# [0] para tirar da lista; interessa-nos as entries 
	print(entries_keys)

'''
resultado do print(entries_keys) para cada site: (igual para todos)

dict_keys(['media_content', 'guidislink', 'title_detail', 'link', 'tags', 'author_detail', 'content', 'author', 'authors', 'title', 'published_parsed', 'summary', 'published', 'links', 'credit', 'id', 'summary_detail', 'media_credit'])
'''

# e para tirar 'content'

'''
with urlopen("https://sunlightlabs.github.io/congress/legislators?api_key='(myapikey)") as conn:
    # dosomething
    a = 0
'''
