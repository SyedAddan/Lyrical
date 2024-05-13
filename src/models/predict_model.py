import os
import re
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

PROJECT_DIR = Path(__file__).resolve().parents[2]

genre_mappings = {
    "pop": "Pop",
    "rock": "Rock",
    "country": "Country",
    "rb": "R&B",
    "rap": "Rap"
}
genres_artists = {'rap': {'21 Savage',
                            '2Pac',
                            '50 Cent',
                            'André 3000',
                            'Big Boi',
                            'Big Daddy Kane',
                            'Big Pun',
                            'Busta Rhymes',
                            'DMX',
                            'Denzel Curry',
                            'Drake',
                            'Eminem',
                            'J. Cole',
                            'JAY-Z',
                            'Jack Harlow',
                            'Kanye WestRedman',
                            'Kendrick Lamar',
                            'Lil Dicky',
                            'Lil Wayne',
                            'Logic',
                            'MF DOOM',
                            'Migos',
                            'Missy Elliott',
                            'Nas',
                            'Nicki Minaj',
                            'OutKast',
                            'Pusha T',
                            'Rakim',
                            'ScHoolboy Q',
                            'Scarface',
                            'Snoop Dogg',
                            'The Notorious B.I.G.',
                            'Travis Scott',
                            'Wu-Tang Clan'},
                    'pop': {'Adele',
                            'Amy Winehouse',
                            'Ariana Grande',
                            'Backstreet Boys',
                            'Barbra Streisand',
                            'Britney Spears',
                            'Bruno Mars',
                            'Charlie Puth',
                            'Christina Aguilera',
                            'Coldplay',
                            'Ed Sheeran',
                            'George Michael',
                            'Janet Jackson',
                            'Jennifer Lopez',
                            'Justin Bieber',
                            'Justin Timberlake',
                            'Katty Perry',
                            'Kelly Clarkson',
                            'Lady Gaga',
                            'Lorde',
                            'Madonna',
                            'Maroon 5',
                            'Michael Jackson',
                            'P!nk',
                            'Rihanna',
                            'Sam Smith',
                            'Selena Gomez',
                            'Shawn Mendes',
                            'Spice Girls',
                            'Stevie Wonder',
                            'StingOne Direction',
                            'Taylor Swift',
                            'The Black Eyed Peas',
                            'Whitney Houston'},
                    'country': {'Alabama',
                                'Alan Jackson',
                                'Brooks & Dunn',
                                'Buck Owens',
                                'Charley Pride',
                                'Chris Stapleton',
                                'Conway Twitty',
                                'Darius Rucker',
                                'Dolly Parton',
                                'Don Williams',
                                'Dwight Yoakam',
                                'Garth Brooks',
                                'George Jones',
                                'George Strait',
                                'Glen Campbell',
                                'Hank Williams',
                                'John Denver',
                                'Johnny Cash',
                                'Keith Urban',
                                'Kenny Rogers',
                                'Loretta Lynn',
                                'Marty Robbins',
                                'Merle Haggard',
                                'Neutral Milk Hotel',
                                'Patsy Cline',
                                'Randy Travis',
                                'Reba McEntire',
                                'Tammy Wynette',
                                'Taylor Swift',
                                'The Chicks',
                                'Toby Keith',
                                'Vince Gill',
                                'Waylon Jennings',
                                'Willie Nelson'},
                    'rb': {'6LACK',
                            'Aaliyah',
                            'Bazzi',
                            'Brent Faiyaz',
                            'Bryson Tiller',
                            'Chase Atlantic',
                            'Chloe x Halle',
                            'Chris Brown',
                            'Ciara',
                            'DPR LIVE',
                            'Daniel Caesar',
                            "Destiny's Child",
                            'Frank Ocean',
                            'H.E.R.',
                            'Jorja Smith',
                            'Kali Uchis',
                            'Kehlani',
                            'Kelela',
                            'Khalid',
                            'Miguel',
                            'Ne-Yo',
                            'Omar Apollo',
                            'PARTYNEXTDOOR',
                            'PinkPantheress',
                            'SZA',
                            'Solange',
                            'Sonder',
                            'Summer Walker',
                            'The Weeknd',
                            'Tinashe',
                            'Tory Lanez',
                            'Tyla',
                            'UMI',
                            'Usher'},
                    'rock': {'3 Doors Down',
                            'Aerosmith',
                            'Audioslave',
                            'Bon Jovi',
                            'Bruce Springsteen',
                            'Coldplay',
                            'David Bowie',
                            'Evanescence',
                            'Foo Fighters',
                            'Garbage',
                            'Incubus',
                            'Jimmy Eat World',
                            'Kings of Leon',
                            'Linkin Park',
                            'Maroon 5',
                            'My Chemical Romance',
                            'Nickelback',
                            'No Doubt',
                            'OneRepublic',
                            'Panic! at the Disco',
                            'Papa Roach',
                            'Paramore',
                            'R.E.M.',
                            'Rage Against the Machine',
                            'Red Hot Chili Peppers',
                            'Seether',
                            'Simple Plan',
                            'Tenacious D',
                            'The Cranberries',
                            'The Police',
                            'The Velvet Underground',
                            'The White Stripes',
                            'U2',
                            'Weezer'}}


# Load the model and tokenizer
model_path = Path(os.path.join(PROJECT_DIR, "models", "gpt2", "model"))
tokenizer_path = Path(os.path.join(PROJECT_DIR, "models", "gpt2", "tokenizer"))

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_lyrics(genre, artist, year, prompt, min_length=200, max_length=600):
    genre_context = f"[s:genre]{genre}[e:genre]"
    artist_context = f"[s:artist]{artist}[e:artist]"
    year_context = f"[s:year]{year}[e:year]"
    lyrics_context = f"[s:lyrics]"
    full_prompt = f"{genre_context} {artist_context} {year_context} {lyrics_context} {prompt}"
    
    input_ids = tokenizer.encode(full_prompt, return_tensors='pt').to(device)

    eos_token_id = tokenizer.encode('[e:lyrics]', add_special_tokens=False)[0]

    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_token_id,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.8
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    clean_text = generated_text.split('[e:lyrics]')[0]  # Strip everything after end lyrics tag
    clean_text = re.sub(r'\[s:genre\].*?\[e:genre\]', '', clean_text)  # Remove genre tags
    clean_text = clean_text.replace(lyrics_context, "").strip()  # Remove lyrics start tag

    return clean_text

def text_pprinter(clean_text):
    clean_text = re.sub(r'\[s:genre\]', '', clean_text)
    clean_text = re.sub(r'\[sgenre\]', '', clean_text) 
    clean_text = re.sub(r'\[egenre\]', '', clean_text) 
    clean_text = re.sub(r'\[e:genre\]', '', clean_text)
    clean_text = re.sub(r'\[sartist\]', '', clean_text) 
    clean_text = re.sub(r'\[s:artist\]', '', clean_text)
    clean_text = re.sub(r'\[eartist\]', '', clean_text) 
    clean_text = re.sub(r'\[e:artist\]', '', clean_text)
    clean_text = re.sub(r'\[syear\]', '', clean_text) 
    clean_text = re.sub(r'\[s:year\]', '', clean_text)
    clean_text = re.sub(r'\[eyear\]', '', clean_text) 
    clean_text = re.sub(r'\[e:year\]', '', clean_text)
    clean_text = re.sub(r'\<|endoftext|>', '', clean_text)
    clean_text = re.sub(r'\||', '', clean_text)
    clean_text = re.sub(r':', '', clean_text)

    print(clean_text)

    
if __name__ == '__main__':  
    genre = input("Enter the genre ('pop', 'rock', 'country', 'rb', 'rap', defaults to 'pop'): ")
    if genre == "":
        genre = "pop"
    genre_mapped = genre_mappings.get(genre, "pop")
    
    artist = input("Enter the artist: ")
    if (artist == "") or (artist not in [artist.lower() for artist in genres_artists[genre]]):
        print(f"Artist not found in the {genre} genre. Defaulting to the first artist in the list")
        artist = list(genres_artists[genre])[0]
        print(f"A random artist from the {genre} genre has been selected: {artist}")
    
    year = input("Enter the year: ")
    if year == "":
        year = 2022
    else:
        year = int(year)
    if not (1950 < year):
        print("Invalid year. Defaulting to 2022")
        year = 2022
    
    prompt = input("Enter the first few lines for the lyrics, or leave empty for the model to decide: ")
    
    generated_lyrics = generate_lyrics(genre, artist, year, prompt)
    
    print()
    print()
    print("--------------------------------------------------------------------")
    print("Generating Lyrics for:")
    print(f"Genre: {genre_mapped}")
    print(f"Artist: {artist}")
    print(f"Year: {year}")
    if prompt:
        print(f"Prompt: {prompt}")
    print("--------------------------------------------------------------------")
    text_pprinter(generated_lyrics)
    print("--------------------------------------------------------------------")