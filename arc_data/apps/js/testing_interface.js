
// Internal state.
var CURRENT_TASK_NAME = ""
var CURRENT_TASK_INDEX = -1;
var CURRENT_INPUT_GRID = new Grid(3, 3);
var CURRENT_OUTPUT_GRID = new Grid(3, 3);
var TEST_PAIRS = new Array();
var CURRENT_TEST_PAIR_INDEX = 0;
var COPY_PASTE_DATA = new Array();
var TASK_FROM_LIST_COUNT = 0;
var TASK_NAME_LIST = new Array();
var TASK_PROGRAM_LIST = new Array();
var EC_OUTPUT = '';
var END_INDEX = 0;

var ITERATIONS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
var ITERATION_INDEX = 0;
var MAX_ITERATION = 0;

var LEARNED_PRODUCTIONS = {};
var TASKS_DICT = {};

// Cosmetic.
var EDITION_GRID_HEIGHT = 500;
var EDITION_GRID_WIDTH = 500;
var MAX_CELL_SIZE = 100;

function resetTask() {
    CURRENT_INPUT_GRID = new Grid(3, 3);
    TEST_PAIRS = new Array();
    CURRENT_TEST_PAIR_INDEX = 0;
    $('#task_preview').html('');
    resetOutputGrid();
}

function get(object, key, default_value) {
    try {
        var result = object[key];
        return result
    } catch (e) {
        return default_value
    }
}

function clearButtonLabels() {
    document.getElementById('random_task').innerHTML = '';
    document.getElementById('task_from_list').innerHTML = '';
    document.getElementById('program_found').innerHTML = '';
}

function loadEcResultFile(e) {
    var file = e.target.files[0];
    if (!file) {
        errorMsg('No file selected');
        return;
    }
    var reader = new FileReader();
    reader.onload = function (e) {
        var contents = e.target.result;
        try {
            var json = JSON.parse(contents)
            TASKS_DICT = json['frontiersOverTime'];
            LEARNED_PRODUCTIONS = json['learnedProductions'];
            MAX_ITERATION = Object.keys(LEARNED_PRODUCTIONS).length
            console.log("MAX_ITERATION", MAX_ITERATION)
            TASK_NAME_LIST = []
            Object.keys(TASKS_DICT).forEach(function(key) {
                if (Object.keys(TASKS_DICT[key]).length > 0) {
                    TASK_NAME_LIST.push(key)
                }
            console.log(TASK_NAME_LIST)
            });
        } catch (e) {
            errorMsg('Bad file format');
            console.log("Error", e.stack);
            console.log("Error", e.name);
            console.log("Error", e.message);
            return;
        }
    }
    reader.readAsText(file);
    console.log(TASKS_DICT);
    TASK_FROM_LIST_COUNT = 0
    return
}



function loadEcOutputFile(e) {
    console.log('called loadEcOutputFile')
    var file = e.target.files[0];
    if (!file) {
        errorMsg('No file selected');
        return;
    }
    var contents = '';
    var reader = new FileReader();
    reader.onload = function (e) {
        EC_OUTPUT = e.target.result;
        try {
            let regex = new RegExp('HIT[^\n]+', 'g')
            if (EC_OUTPUT.includes('enumeration results')) {
                END_INDEX = EC_OUTPUT.indexOf('Average description length of a program solving a task');
                console.log(END_INDEX);
                contents = EC_OUTPUT.slice(EC_OUTPUT.indexOf('enumeration results'), END_INDEX);
                console.log(contents);
            }
            hitList = contents.match(regex)
            TASK_NAME_LIST = hitList.map((element) => element.slice(4, element.search('json') + 4))
            TASK_NAME_LIST = [...new Set(TASK_NAME_LIST)]
            TASK_PROGRAM_LIST = hitList.map((element) => element.slice(element.search('json') + 8).replace(/log/g, '<br> log'))
            console.log('load EC output')
            console.log(TASK_FROM_LIST_COUNT)
            console.log(TASK_NAME_LIST)
            console.log(TASK_PROGRAM_LIST)
            document.getElementById('iteration_name').innerHTML = ITERATION_NAMES[ITERATION_INDEX];
            taskFromList();
        } catch (e) {
            errorMsg('Bad file format');
            console.log("Error", e.stack);
            console.log("Error", e.name);
            console.log("Error", e.message);
            return;
        }
        loadJSONTask(train, test);
    };
    reader.readAsText(file);
}

function changeIteration(next) {
    if (next) {
        ITERATION_INDEX += 1;
    } else {
        ITERATION_INDEX -= 1;
    }
    document.getElementById('myRange').innerHTML = ITERATION_INDEX;
    updateProgram();
    loadJSONTask(train, test);
}

function outputUpdate(vol) {
    console.log(vol);
    ITERATION_INDEX = parseInt(vol);
    updateProgram();
    document.querySelector('#volume').value = vol;
}

function refreshEditionGrid(jqGrid, dataGrid) {
    fillJqGridWithData(jqGrid, dataGrid);
    setUpEditionGridListeners(jqGrid);
    fitCellsToContainer(jqGrid, dataGrid.height, dataGrid.width, EDITION_GRID_HEIGHT, EDITION_GRID_HEIGHT);
    initializeSelectable();
}

function syncFromEditionGridToDataGrid() {
    copyJqGridToDataGrid($('#output_grid .edition_grid'), CURRENT_OUTPUT_GRID);
}

function syncFromDataGridToEditionGrid() {
    refreshEditionGrid($('#output_grid .edition_grid'), CURRENT_OUTPUT_GRID);
}

function getSelectedSymbol() {
    selected = $('#symbol_picker .selected-symbol-preview')[0];
    return $(selected).attr('symbol');
}

function setUpEditionGridListeners(jqGrid) {
    jqGrid.find('.cell').click(function (event) {
        cell = $(event.target);
        symbol = getSelectedSymbol();

        mode = $('input[name=tool_switching]:checked').val();
        if (mode == 'floodfill') {
            // If floodfill: fill all connected cells.
            syncFromEditionGridToDataGrid();
            grid = CURRENT_OUTPUT_GRID.grid;
            floodfillFromLocation(grid, cell.attr('x'), cell.attr('y'), symbol);
            syncFromDataGridToEditionGrid();
        }
        else if (mode == 'edit') {
            // Else: fill just this cell.
            setCellSymbol(cell, symbol);
        }
    });
}

function resizeOutputGrid() {
    size = $('#output_grid_size').val();
    size = parseSizeTuple(size);
    height = size[0];
    width = size[1];

    jqGrid = $('#output_grid .edition_grid');
    syncFromEditionGridToDataGrid();
    dataGrid = JSON.parse(JSON.stringify(CURRENT_OUTPUT_GRID.grid));
    CURRENT_OUTPUT_GRID = new Grid(height, width, dataGrid);
    refreshEditionGrid(jqGrid, CURRENT_OUTPUT_GRID);
}

function resetOutputGrid() {
    syncFromEditionGridToDataGrid();
    CURRENT_OUTPUT_GRID = new Grid(3, 3);
    syncFromDataGridToEditionGrid();
    resizeOutputGrid();
}

function copyFromInput() {
    syncFromEditionGridToDataGrid();
    CURRENT_OUTPUT_GRID = convertSerializedGridToGridObject(CURRENT_INPUT_GRID.grid);
    syncFromDataGridToEditionGrid();
    $('#output_grid_size').val(CURRENT_OUTPUT_GRID.height + 'x' + CURRENT_OUTPUT_GRID.width);
}

function fillPairPreview(pairId, inputGrid, outputGrid) {
    var pairSlot = $('#pair_preview_' + pairId);
    if (!pairSlot.length) {
        // Create HTML for pair.
        pairSlot = $('<div id="pair_preview_' + pairId + '" class="pair_preview" index="' + pairId + '"></div>');
        pairSlot.appendTo('#task_preview');
    }
    var jqInputGrid = pairSlot.find('.input_preview');
    if (!jqInputGrid.length) {
        jqInputGrid = $('<div class="input_preview"></div>');
        jqInputGrid.appendTo(pairSlot);
    }
    var jqOutputGrid = pairSlot.find('.output_preview');
    if (!jqOutputGrid.length) {
        jqOutputGrid = $('<div class="output_preview"></div>');
        jqOutputGrid.appendTo(pairSlot);
    }

    fillJqGridWithData(jqInputGrid, inputGrid);
    fitCellsToContainer(jqInputGrid, inputGrid.height, inputGrid.width, 200, 200);
    fillJqGridWithData(jqOutputGrid, outputGrid);
    fitCellsToContainer(jqOutputGrid, outputGrid.height, outputGrid.width, 200, 200);
}

function loadJSONTask(train, test) {
    resetTask();
    $('#modal_bg').hide();
    $('#error_display').hide();
    $('#info_display').hide();

    for (var i = 0; i < train.length; i++) {
        pair = train[i];
        values = pair['input'];
        input_grid = convertSerializedGridToGridObject(values)
        values = pair['output'];
        output_grid = convertSerializedGridToGridObject(values)
        fillPairPreview(i, input_grid, output_grid);
    }
    for (var i = 0; i < test.length; i++) {
        pair = test[i];
        TEST_PAIRS.push(pair);
    }
    values = TEST_PAIRS[0]['input'];
    CURRENT_INPUT_GRID = convertSerializedGridToGridObject(values)
    fillTestInput(CURRENT_INPUT_GRID);
    CURRENT_TEST_PAIR_INDEX = 0;
    $('#current_test_input_id_display').html('1');
    $('#total_test_input_count_display').html(test.length);
}

function loadTaskFromFile(e) {
    var file = e.target.files[0];
    if (!file) {
        errorMsg('No file selected');
        return;
    }
    var reader = new FileReader();
    reader.onload = function (e) {
        var contents = e.target.result;

        try {
            contents = JSON.parse(contents);
            train = contents['train'];
            test = contents['test'];
        } catch (e) {
            errorMsg('Bad file format');
            return;
        }
        loadJSONTask(train, test);
    };
    reader.readAsText(file);
    document.getElementById('taskName').innerHTML = file.name;
    clearButtonLabels();
}

function nextTaskFromList() {
    TASK_FROM_LIST_COUNT = (TASK_FROM_LIST_COUNT + 1) % TASK_NAME_LIST.length
    taskFromList();
}

function previousTaskFromList() {
    TASK_FROM_LIST_COUNT = (TASK_FROM_LIST_COUNT - 1);
    if (TASK_FROM_LIST_COUNT < 0) {
        TASK_FROM_LIST_COUNT = TASK_NAME_LIST.length-1
    }
    taskFromList();
}

function updateProgram() {
    if  (Object.keys(TASKS_DICT).length > 0) {
        document.getElementById('program_found').innerHTML = get(TASKS_DICT[CURRENT_TASK_NAME], `${ITERATION_INDEX}`, `Not Solved`);
        console.log(`TASK_FROM_LIST_COUNT INDEX ${TASK_FROM_LIST_COUNT}`);
        console.log(`ITERATION INDEX ${ITERATION_INDEX}`);
        document.getElementById('new_primitives').innerHTML = Object.values(get(LEARNED_PRODUCTIONS, `${ITERATION_INDEX}`, `No new primitives found`)).join("</p><p>");
    }

}

function taskFromList() {
    console.log(TASK_FROM_LIST_COUNT)
    console.log(TASK_NAME_LIST)
    console.log(TASK_PROGRAM_LIST)

    clearButtonLabels();
    document.getElementById('task_from_list').innerHTML = (TASK_FROM_LIST_COUNT+1).toString() + ' out of ' + TASK_NAME_LIST.length.toString() + ' tasks solved: ' + TASK_NAME_LIST[TASK_FROM_LIST_COUNT];
    CURRENT_TASK_NAME = TASK_NAME_LIST[TASK_FROM_LIST_COUNT];
    updateProgram();
    var subset = "training";
    $.getJSON("https://api.github.com/repos/fchollet/ARC/contents/data/" + subset, function (tasks) {
        var task = tasks.find(task => task.name == CURRENT_TASK_NAME)
        $.getJSON(task["download_url"], function (json) {
            try {
                train = json['train'];
                test = json['test'];
            } catch (e) {
                errorMsg('Bad file format');
                return;
            }
            loadJSONTask(train, test);
            $('#load_task_file_input')[0].value = "";
            infoMsg("Loaded task training/" + task["name"]);
        })
            .error(function () {
                errorMsg('Error loading task');
            });
        document.getElementById('taskName').innerHTML = task['name'];

    })
        .error(function () {
            errorMsg('Error loading task list');
        });
}


function randomTask() {
    TASK_FROM_LIST_COUNT = 0;
    console.log('task index: ', CURRENT_TASK_INDEX);
    var subset = "training";
    $.getJSON("https://api.github.com/repos/fchollet/ARC/contents/data/" + subset, function (tasks) {

        var CURRENT_TASK_INDEX = document.getElementById('task_index_input').value;
        CURRENT_TASK_INDEX = parseInt(CURRENT_TASK_INDEX);
        console.log("read_from_form parsed", CURRENT_TASK_INDEX);
        console.log("isNaN", isNaN(CURRENT_TASK_INDEX));
        if (isNaN(CURRENT_TASK_INDEX)) {
            console.log('tasks length ', tasks.length)
            CURRENT_TASK_INDEX = Math.floor(Math.random() * tasks.length);    
        }

        console.log('final ', CURRENT_TASK_INDEX);
        CURRENT_TASK_NAME = tasks[CURRENT_TASK_INDEX].name;
        $.getJSON(tasks[CURRENT_TASK_INDEX]["download_url"], function (json) {
            try {
                train = json['train'];
                test = json['test'];
            } catch (e) {
                errorMsg('Bad file format');
                return;
            }
            loadJSONTask(train, test);
            $('#load_task_file_input')[0].value = "";
            infoMsg("Loaded task training/" + CURRENT_TASK_NAME);
        })
            .error(function () {
                errorMsg('Error loading task');
            });
        clearButtonLabels();
        document.getElementById('taskName').innerHTML = `Task ${CURRENT_TASK_INDEX} out of ${tasks.length}: ${CURRENT_TASK_NAME}`;
        document.getElementById('random_task').innerHTML = `Task ${CURRENT_TASK_INDEX} out of ${tasks.length}: ${CURRENT_TASK_NAME}`;
        document.getElementById('task_from_list').innerHTML = `Task ${CURRENT_TASK_INDEX} out of ${tasks.length}: ${CURRENT_TASK_NAME}`;
        document.getElementById('taskName').innerHTML = `Task ${CURRENT_TASK_INDEX} out of ${tasks.length}: ${CURRENT_TASK_NAME}`;
        updateProgram();
    })
        .error(function () {
            errorMsg('Error loading task list');
        });
}

function nextTestInput() {
    if (TEST_PAIRS.length <= CURRENT_TEST_PAIR_INDEX + 1) {
        errorMsg('No next test input. Pick another file?')
        return
    }
    CURRENT_TEST_PAIR_INDEX += 1;
    values = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['input'];
    CURRENT_INPUT_GRID = convertSerializedGridToGridObject(values)
    fillTestInput(CURRENT_INPUT_GRID);
    $('#current_test_input_id_display').html(CURRENT_TEST_PAIR_INDEX + 1);
    $('#total_test_input_count_display').html(test.length);
}

function submitSolution() {
    syncFromEditionGridToDataGrid();
    reference_output = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['output'];
    submitted_output = CURRENT_OUTPUT_GRID.grid;
    if (reference_output.length != submitted_output.length) {
        errorMsg('Wrong solution.');
        return
    }
    for (var i = 0; i < reference_output.length; i++) {
        ref_row = reference_output[i];
        for (var j = 0; j < ref_row.length; j++) {
            if (ref_row[j] != submitted_output[i][j]) {
                errorMsg('Wrong solution.');
                return
            }
        }

    }
    infoMsg('Correct solution!');
}

function fillTestInput(inputGrid) {
    jqInputGrid = $('#evaluation_input');
    fillJqGridWithData(jqInputGrid, inputGrid);
    fitCellsToContainer(jqInputGrid, inputGrid.height, inputGrid.width, 400, 400);
}

function copyToOutput() {
    syncFromEditionGridToDataGrid();
    CURRENT_OUTPUT_GRID = convertSerializedGridToGridObject(CURRENT_INPUT_GRID.grid);
    syncFromDataGridToEditionGrid();
    $('#output_grid_size').val(CURRENT_OUTPUT_GRID.height + 'x' + CURRENT_OUTPUT_GRID.width);
}

function initializeSelectable() {
    try {
        $('.selectable_grid').selectable('destroy');
    }
    catch (e) {
    }
    toolMode = $('input[name=tool_switching]:checked').val();
    if (toolMode == 'select') {
        infoMsg('Select some cells and click on a color to fill in, or press C to copy');
        $('.selectable_grid').selectable(
            {
                autoRefresh: false,
                filter: '> .row > .cell',
                start: function (event, ui) {
                    $('.ui-selected').each(function (i, e) {
                        $(e).removeClass('ui-selected');
                    });
                }
            }
        );
    }
}

// Initial event binding.

$(document).ready(function () {
    $('#symbol_picker').find('.symbol_preview').click(function (event) {
        symbol_preview = $(event.target);
        $('#symbol_picker').find('.symbol_preview').each(function (i, preview) {
            $(preview).removeClass('selected-symbol-preview');
        })
        symbol_preview.addClass('selected-symbol-preview');

        toolMode = $('input[name=tool_switching]:checked').val();
        if (toolMode == 'select') {
            $('.edition_grid').find('.ui-selected').each(function (i, cell) {
                symbol = getSelectedSymbol();
                setCellSymbol($(cell), symbol);
            });
        }
    });

    $('.edition_grid').each(function (i, jqGrid) {
        setUpEditionGridListeners($(jqGrid));
    });

    $('.load_task').on('change', function (event) {
        loadTaskFromFile(event);
    });

    $('.load_task').on('click', function (event) {
        event.target.value = "";
    });

    $('.load_ec_output').on('change', function (event) {
        loadEcResultFile(event);
    });

    $('.load_ec_output').on('click', function (event) {
        event.target.value = "";
    });

    $('input[type=radio][name=tool_switching]').change(function () {
        initializeSelectable();
    });

    $('body').keydown(function (event) {
        // Copy and paste functionality.
        if (event.which == 67) {
            // Press C

            selected = $('.ui-selected');
            if (selected.length == 0) {
                return;
            }

            COPY_PASTE_DATA = [];
            for (var i = 0; i < selected.length; i++) {
                x = parseInt($(selected[i]).attr('x'));
                y = parseInt($(selected[i]).attr('y'));
                symbol = parseInt($(selected[i]).attr('symbol'));
                COPY_PASTE_DATA.push([x, y, symbol]);
            }
            infoMsg('Cells copied! Select a target cell and press V to paste at location.');

        }
        if (event.which == 86) {
            // Press P
            if (COPY_PASTE_DATA.length == 0) {
                errorMsg('No data to paste.');
                return;
            }
            selected = $('.edition_grid').find('.ui-selected');
            if (selected.length == 0) {
                errorMsg('Select a target cell on the output grid.');
                return;
            }

            jqGrid = $(selected.parent().parent()[0]);

            if (selected.length == 1) {
                targetx = parseInt(selected.attr('x'));
                targety = parseInt(selected.attr('y'));

                xs = new Array();
                ys = new Array();
                symbols = new Array();

                for (var i = 0; i < COPY_PASTE_DATA.length; i++) {
                    xs.push(COPY_PASTE_DATA[i][0]);
                    ys.push(COPY_PASTE_DATA[i][1]);
                    symbols.push(COPY_PASTE_DATA[i][2]);
                }

                minx = Math.min(...xs);
                miny = Math.min(...ys);
                for (var i = 0; i < xs.length; i++) {
                    x = xs[i];
                    y = ys[i];
                    symbol = symbols[i];
                    newx = x - minx + targetx;
                    newy = y - miny + targety;
                    res = jqGrid.find('[x="' + newx + '"][y="' + newy + '"] ');
                    if (res.length == 1) {
                        cell = $(res[0]);
                        setCellSymbol(cell, symbol);
                    }
                }
            } else {
                errorMsg('Can only paste at a specific location; only select *one* cell as paste destination.');
            }
        }
    });
});
