import os
import re
import pandas as pd
from pathlib import Path

genre_mappings = {
    "pop": "Pop",
    "rock": "Rock",
    "country": "Country",
    "rb": "R&B",
    "rap": "Rap"
}

PROJECT_DIR = Path(__file__).resolve().parents[2]


def clean_lyric_content(lyric_text):
    try:
        contributor_pattern = re.compile(
            r'\d+ ContributorsTranslations.*?Lyrics')
        bracket_pattern = re.compile(r'\[.*?\](?!\[s:|\[e:)')
        ascii_pattern = re.compile(r'[^\x00-\x7F]+')
        newline_pattern = re.compile(r'\n{3,}')

        # Apply regex patterns
        lyric_text = re.sub(contributor_pattern, '', lyric_text)
        lyric_text = re.sub(bracket_pattern, '', lyric_text)
        lyric_text = re.sub(ascii_pattern, ' ', lyric_text)
        lyric_text = re.sub(newline_pattern, '\n\n', lyric_text)

        # Removing quotation marks
        lyric_text = lyric_text.replace('"', '')

        return lyric_text.strip()
    except:
        return ""


def lyrics_prompter(lyrics, genre, artist, year):
    prompted_lyric = f'[s:genre]{genre_mappings[genre]}[e:genre]' + ' ' + f'[s:artist]{artist}[e:artist]' + \
        ' ' + f'[s:year]{year}[e:year]' + " " + \
        f'[s:lyrics] {lyrics} [e:lyrics]' + '\n'
    return prompted_lyric


def process_lyrics_file(input_filepath, output_filepath, genres_artists):
    # Read CSV file
    df = pd.read_csv(input_filepath)

    # Check for required columns
    if not all(col in df.columns for col in ['artist', 'tag', 'views', 'lyrics', 'year']):
        raise ValueError("Required columns are missing from the input file")
    else:
        df = df[['title', 'artist', 'tag', 'year', 'views', 'lyrics']]

    final_lyrics_df = pd.DataFrame(columns=[
                                   'title', 'artist', 'tag', 'year', 'views', 'lyrics', "cleaned_lyrics", "final_lyrics"])

    for genre, artists in genres_artists.items():
        genre_df = df[df['tag'].str.lower() == genre.lower()]
        for artist in artists:
            artist_df = genre_df[genre_df['artist'].str.contains(
                artist, case=False, regex=False)]
            top_songs = artist_df.nlargest(min(300, len(artist_df)), 'views')

            # Clean lyrics
            top_songs['cleaned_lyrics'] = top_songs['lyrics'].apply(
                clean_lyric_content)
            top_songs = top_songs.dropna()

            # Prepare tags and combine
            top_songs['final_lyrics'] = top_songs.apply(lambda x: lyrics_prompter(
                x["cleaned_lyrics"], genre, artist, x["year"]), axis=1)

            final_lyrics_df = pd.concat(
                [final_lyrics_df, top_songs], ignore_index=True)

    final_lyrics_df.to_csv(output_filepath, index=False)


if __name__ == "__main__":

    # File paths
    input_filepath = Path(os.path.join(
        PROJECT_DIR, "data", "processed", "lyrics_processed.csv"))
    output_filepath = Path(os.path.join(
        PROJECT_DIR, "data", "features", "lyrics_features.csv"))
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

    # Process lyrics
    process_lyrics_file(input_filepath, output_filepath, genres_artists)
