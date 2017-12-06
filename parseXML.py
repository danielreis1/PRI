from urllib.request import urlopen
import feedparser
import re
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
clean_tags = re.compile(r'<[^>]+>')	# regular expression to clean HTML tags from text

def remove_tags(text):
	return clean_tags.sub('', text)

for site in sites:
	print('\n\t***\t' + site + '\t***\t')
	d[site] = feedparser.parse(site)
	entries = d[site]['entries']

	for entry in entries:
		#links = {site : entry['link']}
		titles_source = {entry['link'] : entry['title']}
		titles = list(titles_source.values())[0]
		summary = remove_tags(re.sub('\.{3}', '', entry['summary'])) # tira tags HTML e reticencias
		summaries_source = {entry['link'] : summary}
		summaries = list(summaries_source.values())[0]
		print('\n***\tTitle\t***')
		print(titles)
		print('\n***\tSummary\t***')
		print(summaries)

