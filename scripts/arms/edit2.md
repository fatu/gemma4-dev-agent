mode: replace
anchor: Copy old_string exactly from the source, including whitespace, and choose a short unique match.
---
Right before each edit_file, read the lines you will change again (read_file with start_line and end_line, about 20 lines) and copy old_string character for character from that latest output, never from memory and never from your new_string. Use 1-3 whole lines that contain a name unique in the file, such as the def or class line or a distinctive identifier.
