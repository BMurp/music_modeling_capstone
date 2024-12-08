#functions generating inputs to HTML template
import matplotlib.pyplot as plt
import librosa
import librosa.display
import base64
from io import BytesIO
import numpy as np




def get_log_mel_spectrogram_image_base_64(log_mel_s_):
    '''for given log melspectogram nd array, return base64 encoded string of librosa specshow heatmap'''
    fig, ax = plt.subplots()
    spectro = librosa.display.specshow(log_mel_s_,x_axis='time', y_axis='mel', fmax=8000)
    fig.colorbar(spectro, ax=ax)
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    plt.close()
    return encoded


def get_predicted_probs_stacked_bar_image_base_64(predicted_probs, label_names):
    #referenced from #https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html

    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    results = {
    'model 1': [round(value*100,0) for value in predicted_probs],
    #'model 2': final_values

    }
    #category_names = final_labels
    category_names = label_names

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))
    category_colors = plt.colormaps['twilight_shifted'](
        np.linspace(0.15, 0.85, data.shape[1]))
    #print(category_colors)

    #fig, ax = plt.subplots(figsize=(9.2, 5))
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        if widths >=1:
            rects = ax.barh(labels, widths, left=starts, height=0.5,
                            label=colname, color=color)

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    
    tmpfile = BytesIO()
    #fig.savefig(tmpfile, format='png')
    plt.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    plt.close()

    return encoded



def get_image_html(encoded_image):
    '''for given base64 encoded image return html image tag'''
    return f'''<img src="data:image/png;base64,{encoded_image}" alt="alt image width="100" height="100">'''

def get_audio_player_html(audio_path):
    '''returns html audio player for audio in a given local path'''
    audio_html = f'''<audio controls autoplay><source src="{audio_path}" type="audio/mpeg">Your browser does not support the audio element.</audio>'''
    return audio_html

def get_log_mel_spectrogram_html_image_array(log_melspectroram_array):
    '''for provided array of log_spectrograms, returns array of html images'''
    plt.ioff()
    html_images = []
    for lm in log_melspectroram_array:
        encoded = get_log_mel_spectrogram_image_base_64(lm)
        html = get_image_html(encoded)
        html_images.append(html)
    return html_images

def get_predicted_probs_stacked_bar_html_image_array(predicted_probs_df):
    '''for provided array of log_spectrograms, returns array of html images'''
    label_names = predicted_probs_df.columns
    plt.ioff()
    html_images = []
    for row in list(range(0, len(predicted_probs_df))):
        encoded = get_predicted_probs_stacked_bar_image_base_64(predicted_probs_df.iloc[row].values, label_names)
        html = get_image_html(encoded)
        html_images.append(html)
    return html_images

def get_audio_player_html_array(audio_paths):
    '''for provided array of audio_paths, return array of html audio players'''
    html_players = []
    for path in audio_paths:
        html_players.append(get_audio_player_html(path))
    return html_players


def get_table_rows(audio_browser_table_data):
    '''returns html for all rows in a table according to audio_browser_table_data dictionary
    expects dictionary keys to be column names, and arrays of html elements as values
    '''
    table_data_list = list(audio_browser_table_data.items())
    num_columns = len(table_data_list)
    num_rows = len(table_data_list[0][1])
    #print('num rows: ', num_rows)
    all_rows = ''
    for row_index in list(range(0,num_rows)):
       # print('row: ', row_index)
        current_row_html = "<tr>\n"
        for column_index in list(range(0,num_columns)):
            #print('column: ', column_index)
            current_row_html += "<td>" + table_data_list[column_index][1][row_index]+"</td>"
            current_row_html += "\n"
        current_row_html += "</tr>\n"
            
        all_rows += current_row_html 
    return all_rows

def get_audio_browser_html(audio_browser_table_data):
    audio_browser_header = '''
    <html>
    <head>
        <title>Table with Images</title>
        <style>
            table {{
                margin: auto;
            }}
        </style>
    </head>

    <body>
        <table border="1">
    '''
    #header for each of the keys in provided dic
    table_headers = "<tr>\n" + ''.join(f'<th>{key}</th>\n' for key in list(audio_browser_table_data.keys())) +"</tr>\n"
    audio_browser_footer = '''
        </table>
    </body>
    </html>
    '''
    #get the rows of table
    table_rows = get_table_rows(audio_browser_table_data)
    #assembles the page 
    table_html = audio_browser_header+table_headers+table_rows+audio_browser_footer
    return table_html