mode: replace
anchor: 3. Edit with edit_file. For old_string, copy 2-5 whole lines exactly as read_file showed them, so the match is unique. If edit_file fails twice, read that region again and use a smaller exact snippet.
---
3. Edit with edit_file. For old_string, copy ONE whole line character for character from the latest read_file output of that file: the line you change, or the line just above where new code goes. Never retype it from memory. new_string replaces only that line, so it holds the full new line, or that line again followed by the inserted lines. Add the next line to old_string only if edit_file reports several matches. If edit_file says old_string not found, read_file those lines again and copy one line from the new output.
