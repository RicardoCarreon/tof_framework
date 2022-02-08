The code contained in the files "msel_utilities.js" and "custom.js" create custom Jupyter notebook buttons to manipulate the magic selectors. 
To activate them, the files have to be copied to the custom directory of the jupyter installation. This is usually:

C:\Users\<username>\.jupyter\custom\custom.js

The "custom.js" file and the "custom" folder does not necessarily exist. If not, create the "custom folder" and copy both files there.
If the "custom.js" file already exists at that location, copy the "msel_utilities.js" files there and add the line of code from "custom.js"
in the already existing "custom.js" file of the Jupyter installation folder.

If it doesn't work, try to run the following command "echo $(jupyter --config-dir)/custom/custom.js" in a terminal, which will give the
correct path for the "custom.js" file.