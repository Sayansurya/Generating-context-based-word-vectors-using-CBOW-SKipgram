import re
import traceback
import json

from bs4 import BeautifulSoup as BS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

WORDS = []
K = 10

with open('../data/validation.txt', 'r') as f:
    analogy_dataset = f.readlines()
for sample in analogy_dataset:
    WORDS.extend(sample.strip().split())

HOMEPAGE = 'https://en.wikipedia.org/w/index.php?search'
driver = webdriver.Chrome(
    ChromeDriverManager().install()
)


def sent_processor(text):  # REF => https://stackoverflow.com/questions/37528373/how-to-remove-all-text-between-the-outer-parentheses-in-a-string
    n = 1
    while n:
        text, n = re.subn(r'\([^()]*\)', '', text)
    text = re.sub(r'[\[].*?[\]]', '', text)
    return text


def get_k_sentences(word):
    driver.get(HOMEPAGE)
    search = WebDriverWait(driver, 3).until(
        EC.presence_of_element_located((By.ID, "ooui-php-1"))
    )
    search.clear()
    search.send_keys(word)
    search.send_keys(Keys.ENTER)
    show_100 = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located(
            (By.XPATH, "//a[@title='Show 500 results per page']")
        )
    )
    show_100.click()
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.CLASS_NAME, "mw-search-results-container"))
    )
    soup = BS(driver.page_source, "html.parser")
    result_body = soup.find("div", {"class": "mw-search-results-container"})
    suggestion_links = result_body.find_all("a")
    k_sent = []
    for link in tqdm(suggestion_links, desc=f'checking for "{word}" in pages:- '):
        driver.get("https://en.wikipedia.org"+link.get('href'))
        WebDriverWait(driver, 3).until(
            EC.presence_of_element_located(
                (By.XPATH, "//div[@class='mw-parser-output']"))
        )
        p_tags = driver.find_elements(
            By.XPATH, "//div[@class='mw-parser-output']/p")
        for paragraph in p_tags:
            sentences = paragraph.text.split(". ")
            for sentence in sentences:
                if word.lower() in sentence.lower():
                    if not sentence.endswith("."):
                        sentence += "."
                    k_sent.append(sent_processor(sentence))
                    if len(k_sent) == K:
                        return k_sent
    return k_sent


with open('../data/scraped_sentences_validation.json', 'r', encoding='utf-8') as f:
    scraped_sentences = json.load(f)

WORDS = set(WORDS)

for word in tqdm(WORDS, desc="processing words"):
    try:
        if word not in scraped_sentences.keys():
            sentences = get_k_sentences(word)
            print(f'For {word} scraped {len(sentences)} sentences.')
            scraped_sentences[word] = sentences
            with open('../data/scraped_sentences_validation.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(scraped_sentences,
                        indent=2, ensure_ascii=False))
                print('saved file.')
    except:
        print(f'ERROR for "{word}": \n{traceback.format_exc()}')

with open('../data/scraped_sentences_validation.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(scraped_sentences, indent=2, ensure_ascii=False))
    print('saved file.')
