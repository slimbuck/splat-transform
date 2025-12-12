import { open } from 'node:fs/promises';
import { DataTable } from './data-table';
import { logger } from './logger';
import { isCompressedPly, decompressPly } from './readers/decompress-ply';
import { readKsplat } from './readers/read-ksplat';
import { readLcc } from './readers/read-lcc';
import { readMjs } from './readers/read-mjs';
import { readPly } from './readers/read-ply';
import { readSog } from './readers/read-sog';
import { readSplat } from './readers/read-splat';
import { readSpz } from './readers/read-spz';
import { Options, Param } from './types';

type InputFormat = 'mjs' | 'ksplat' | 'splat' | 'sog' | 'ply' | 'spz' | 'lcc';

const getInputFormat = (filename: string): InputFormat => {
    const lowerFilename = filename.toLowerCase();

    if (lowerFilename.endsWith('.mjs')) {
        return 'mjs';
    } else if (lowerFilename.endsWith('.ksplat')) {
        return 'ksplat';
    } else if (lowerFilename.endsWith('.splat')) {
        return 'splat';
    } else if (lowerFilename.endsWith('.sog') || lowerFilename.endsWith('meta.json')) {
        return 'sog';
    } else if (lowerFilename.endsWith('.ply')) {
        return 'ply';
    } else if (lowerFilename.endsWith('.spz')) {
        return 'spz';
    } else if (lowerFilename.endsWith('.lcc')) {
        return 'lcc';
    }

    throw new Error(`Unsupported input file type: ${filename}`);
};

const readFile = async (filename: string, options: Options, params: Param[]): Promise<DataTable[]> => {
    const inputFormat = getInputFormat(filename);
    let result: DataTable[];

    logger.info(`reading '${filename}'...`);

    if (inputFormat === 'mjs') {
        result = [await readMjs(filename, params)];
    } else {
        const inputFile = await open(filename, 'r');

        if (inputFormat === 'ksplat') {
            result = [await readKsplat(inputFile)];
        } else if (inputFormat === 'splat') {
            result = [await readSplat(inputFile)];
        } else if (inputFormat === 'sog') {
            result = [await readSog(inputFile, filename)];
        } else if (inputFormat === 'ply') {
            const ply = await readPly(inputFile);
            if (isCompressedPly(ply)) {
                result = [decompressPly(ply)];
            } else {
                if (ply.elements.length !== 1 || ply.elements[0].name !== 'vertex') {
                    throw new Error(`Unsupported data in file '${filename}'`);
                }
                result = [ply.elements[0].dataTable];
            }
        } else if (inputFormat === 'spz') {
            result = [await readSpz(inputFile)];
        } else if (inputFormat === 'lcc') {
            result = await readLcc(inputFile, filename, options);
        }

        await inputFile.close();
    }

    return result;
};

export { type InputFormat, getInputFormat, readFile };
