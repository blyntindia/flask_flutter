import pandas as pd
from metaphone import doublemetaphone

# Load Excel file
df = pd.read_excel("Database_Edited.xlsx")
df.dropna(subset=['Place_Available', 'Place_Location'], inplace=True)

# Take search keyword input
search_input = input("Enter search keyword: ").strip().lower()

# Identify all tag columns (e.g., Tag_1, Tag_2, etc.)
tag_columns = [col for col in df.columns if col.startswith('Tag_')]

# Convert word to phonetic (metaphone) set
def get_metaphones(word):
    return set(filter(None, doublemetaphone(word)))

search_codes = get_metaphones(search_input)

# Check if any tag sounds similar to the input
def match_phonetic(tags):
    for tag in tags:
        if pd.notna(tag):
            tag_codes = get_metaphones(str(tag).strip().lower())
            if not search_codes.isdisjoint(tag_codes):
                return True
    return False

# Combine tag columns into a single list for each row
df['Matching_Tags'] = df[tag_columns].apply(
    lambda row: [tag for tag in row if pd.notna(tag)],
    axis=1
)

# Apply phonetic matching
df['Phonetic_Match'] = df['Matching_Tags'].apply(match_phonetic)

# Filter matching places
matched_df = df[df['Phonetic_Match']]

# Show results
if not matched_df.empty:
    print(f"\nMatching places for pronunciation of '{search_input}':\n")
    for _, row in matched_df.iterrows():
        tags = ", ".join(str(tag) for tag in row['Matching_Tags'])
        print(f"- {row['Place_Available']} ({row['Place_Location']}) | Tags: {tags}")
else:
    print("\nNo phonetically matching places found.")
