import React, { useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';
import type { ColDef, GridReadyEvent } from 'ag-grid-community';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import type { ExtractedData } from '../../lib/appTypes';
import styles from './DataGrid.module.css';
import { AlertTriangle } from 'lucide-react';

interface DataGridProps {
    data: ExtractedData;
    highlightedRowIndex?: number;
}

export const DataGrid: React.FC<DataGridProps> = ({ data, highlightedRowIndex }) => {

    const columnDefs = useMemo<ColDef[]>(() => {
        return data.columns.map(col => ({
            field: col.field,
            headerName: col.headerName,
            editable: true,
            sortable: true,
            filter: true,
            resizable: true,
            cellRenderer: (params: any) => {
                // Check for errors in this cell
                const error = data.errors?.find(
                    e => e.row === params.node.rowIndex && e.column === col.field
                );

                if (error) {
                    return (
                        <div className={styles.errorCell} title={error.reason}>
                            <span className={styles.cellValue}>{params.value}</span>
                            <AlertTriangle size={14} className={styles.errorIcon} />
                        </div>
                    );
                }
                return params.value;
            }
        }));
    }, [data]);

    const defaultColDef = useMemo(() => ({
        flex: 1,
        minWidth: 100,
    }), []);

    const onGridReady = (params: GridReadyEvent) => {
        params.api.sizeColumnsToFit();
    };

    // Highlight row effect
    const getRowStyle = (params: any) => {
        if (params.node.rowIndex === highlightedRowIndex) {
            return { background: 'rgba(37, 99, 235, 0.1)' };
        }
        return undefined;
    };

    return (
        <div className={`ag-theme-alpine ${styles.gridContainer}`}>
            <AgGridReact
                rowData={data.rows}
                columnDefs={columnDefs}
                defaultColDef={defaultColDef}
                onGridReady={onGridReady}
                getRowStyle={getRowStyle}
                animateRows={true}
                rowSelection="single"
            />
        </div>
    );
};
