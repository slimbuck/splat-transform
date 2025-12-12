import json from '@rollup/plugin-json';
import resolve from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';

const library = {
    input: 'src/index.ts',
    output: {
        dir: 'dist',
        format: 'esm',
        sourcemap: true,
        entryFileNames: '[name].mjs'
    },
    external: [
        'webgpu',
        'playcanvas',
        /^node:/,
        'crypto'
    ],
    plugins: [
        typescript({
            tsconfig: false,
            compilerOptions: {
                target: 'es2022',
                module: 'es2022',
                lib: ['es2022', 'dom'],
                moduleResolution: 'bundler',
                esModuleInterop: true,
                sourceMap: true,
                resolveJsonModule: true
            },
            include: ['src/**/*.ts']
        }),
        resolve({
            extensions: ['.mjs', '.js', '.json', '.node']
        }),
        json()
    ],
    cache: false
};

export default [
    library
];
