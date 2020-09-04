from googletrans import Translator, constants

import warnings
warnings.filterwarnings("ignore")
class TweetTranslator:
	def __init__(self):
		"""
		Class used for translating tweets from english to romanian and reversed
		"""
		pass
	def translate(self, text, first_lang, second_lang):
		"""
		Parameters
		----------
		text : string
			the tweet that will be translated
		first_lang : string
			the source language of the tweet
		second_lang : string
			the destination language of the tweet

		Returns
		----------
		translation.text : string
			the translation of the string text from the
			language first_lang to the language second_lang
		"""
		translator = Translator()
		if first_lang=='ro' and second_lang=='en':
			translation = translator.translate(text)
			return translation.text
		elif first_lang=='en' and second_lang=='ro':
			translation = translator.translate(text, dest=second_lang)
			return translation.text



