for f in *.svg; do inkscape "$f" \
  --export-type=png --export-filename="${f%.svg}.png" \
  --export-width=2560 --export-area-page; done

