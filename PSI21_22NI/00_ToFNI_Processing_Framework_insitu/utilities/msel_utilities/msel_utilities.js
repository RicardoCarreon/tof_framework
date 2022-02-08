define([
         'base/js/namespace',
         'base/js/promises'
     ], function(IPython, promises) {
         promises.app_initialized.then(function (appName) {
             if (appName !== 'NotebookApp') return;
             IPython.toolbar.add_buttons_group([
                 
                 {
                 
                     'label'   : 'Magic Selectors:',
                     'icon'    : 'fa-magic'
                     
                 },
                 {
                 
                         // this is a button to reset all the magic selector cells
                     'label'   : 'Lock',
                     'icon'    : 'fa-lock',
                     'callback': function () {
        					
                					// loop for all selected cells
                            var cells = Jupyter.notebook.get_selected_cells()
                            for (i=0; i<cells.length; i++) {
                            
                        				// get the cell text and split it in lines
                                var cell_lines = cells[i].get_text().split('\n')
                                var count = cell_lines.length
                                    
                                    // if the last line starts with "# %msel"
                                    // this is a magic selector cell which needs to be locked
                                if (cell_lines[count-1].startsWith("# %msel")) {
                                
                                        // add a '#' sign to the last line
                                    cell_lines[count-1] = '#' + cell_lines[count-1]
                                    cells[i].set_text(cell_lines.join('\n'))
                                }
                                
                            }
                     }
                 },
                 
                 
                 {
                 
                         // this is a button to reset all the magic selector cells
                     'label'   : 'Unlock',
                     'icon'    : 'fa-unlock',
                     'callback': function () {
        					
                					// loop for all selected cells
                            var cells = Jupyter.notebook.get_selected_cells()
                            for (i=0; i<cells.length; i++) {
                            
                        				// get the cell text and split it in lines
                                var cell_lines = cells[i].get_text().split('\n')
                                var count = cell_lines.length
                                    
                                    // if the last line starts with "## %msel"
                                    // this is a magic selector cell which needs to be unlocked
                                if (cell_lines[count-1].startsWith("## %msel")) {
                                
                                        // remove a '#' sign from the last line
                                    cell_lines[count-1] = cell_lines[count-1].substring(1)
                                    cells[i].set_text(cell_lines.join('\n'))
                                }
                                
                            }
                     }
                 },
                 
                 
                 {
                 
                         // this is a button to reset all the magic selector cells
                     'label'   : 'Reset',
                     'icon'    : 'fa-undo',
                     'callback': function () {
        					
                					// loop for all selected cells
                            var cells = Jupyter.notebook.get_selected_cells()
                            for (i=0; i<cells.length; i++) {
                            
                        				// get the cell text and split it in lines
                                var cell_lines = cells[i].get_text().split('\n')
                                var count = cell_lines.length
                                    
                                    // if the last line starts with "## %msel"
                                    // this is a magic selector cell which needs to be unlocked
                                if (cell_lines[count-1].startsWith("# %msel")) {
                            
                                    // keep only the last line without the two first characters
                                var new_text = cell_lines[count-1].substring(2)
                                cells[i].set_text(new_text)
                                }
                                
                            }
                     }
                 },
                 
                 {
                 
                         // this is a button to reset all the magic selector cells
                     'label'   : 'Reset All',
                     'icon'    : 'fa-bomb',
                     'callback': function () {
                     
                             // ask for confirmation
                        if (confirm('Are you sure ? All your interactive inputs will be erased !')) {
        					
            					// loop for all cells in the notebook
        					var i = 0
        					while (Jupyter.notebook.get_cell(i) != null) {
        					
                					// get the cell text and split it in lines
                            var cell_text = Jupyter.notebook.get_cell(i).get_text()
                            var cell_lines = cell_text.split('\n')
                            var count = cell_lines.length
                                
                                // if the last line starts with "# %msel"
                                // this is a magic selector cell which needs to be reset
                            if (cell_lines[count-1].startsWith("# %msel")) {
                            
                                    // keep only the last line without the two first characters
                                var new_text = cell_lines[count-1].substring(2)
                                Jupyter.notebook.get_cell(i).set_text(new_text)
                            } i++ } }
                     }
                 }
                // add more button here if needed.
                 ]);
         });
     });
