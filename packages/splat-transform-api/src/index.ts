// Core data structures
export { Column, DataTable, TypedArray } from './data-table';

// Types
export type { Options, Param } from './types';

// Reading and writing
export { type InputFormat, getInputFormat, readFile } from './read';
export { type OutputFormat, getOutputFormat, writeFile } from './write';

// Processing and transformations
export { ProcessAction, processDataTable } from './process';
export { transform } from './transform';

// Utilities
export { logger } from './logger';

// Ordering
export { generateOrdering } from './ordering';

// GPU utilities
export { enumerateAdapters } from './gpu/gpu-device';

// Utility functions extracted from CLI for reuse

import { Column, DataTable, TypedArray } from './data-table';

/**
 * Combine multiple DataTables into one.
 * Columns with matching name and type are combined.
 */
const combine = (dataTables: DataTable[]): DataTable => {
    if (dataTables.length === 1) {
        // nothing to combine
        return dataTables[0];
    }

    const findMatchingColumn = (columns: Column[], column: Column) => {
        for (let i = 0; i < columns.length; ++i) {
            if (columns[i].name === column.name &&
                columns[i].dataType === column.dataType) {
                return columns[i];
            }
        }
        return null;
    };

    // make unique list of columns where name and type much match
    const columns = dataTables[0].columns.slice();
    for (let i = 1; i < dataTables.length; ++i) {
        const dataTable = dataTables[i];
        for (let j = 0; j < dataTable.columns.length; ++j) {
            if (!findMatchingColumn(columns, dataTable.columns[j])) {
                columns.push(dataTable.columns[j]);
            }
        }
    }

    // count total number of rows
    const totalRows = dataTables.reduce((sum, dataTable) => sum + dataTable.numRows, 0);

    // construct output dataTable
    const resultColumns = columns.map((column) => {
        const constructor = column.data.constructor as new (length: number) => TypedArray;
        return new Column(column.name, new constructor(totalRows));
    });
    const result = new DataTable(resultColumns);

    // copy data
    let rowOffset = 0;
    for (let i = 0; i < dataTables.length; ++i) {
        const dataTable = dataTables[i];

        for (let j = 0; j < dataTable.columns.length; ++j) {
            const column = dataTable.columns[j];
            const targetColumn = findMatchingColumn(result.columns, column);
            targetColumn.data.set(column.data, rowOffset);
        }

        rowOffset += dataTable.numRows;
    }

    return result;
};

/**
 * Check if a DataTable contains the required columns for a Gaussian Splat.
 */
const isGSDataTable = (dataTable: DataTable): boolean => {
    if (![
        'x', 'y', 'z',
        'rot_0', 'rot_1', 'rot_2', 'rot_3',
        'scale_0', 'scale_1', 'scale_2',
        'f_dc_0', 'f_dc_1', 'f_dc_2',
        'opacity'
    ].every(c => dataTable.hasColumn(c))) {
        return false;
    }
    return true;
};

export { combine, isGSDataTable };

