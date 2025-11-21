import React from 'react';
import { FileSpreadsheet, FileJson, FileText } from 'lucide-react';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import type { ExtractedData } from '../../lib/appTypes';
import styles from './ExportButtons.module.css';

interface ExportButtonsProps {
    data: ExtractedData;
}

export const ExportButtons: React.FC<ExportButtonsProps> = ({ data }) => {

    const handleExportCSV = () => {
        const csv = Papa.unparse(data.rows);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'extracted_data.csv';
        link.click();
    };

    const handleExportXLSX = () => {
        const ws = XLSX.utils.json_to_sheet(data.rows);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, 'Data');
        XLSX.writeFile(wb, 'extracted_data.xlsx');
    };

    const handleExportJSON = () => {
        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'extracted_data.json';
        link.click();
    };

    return (
        <div className={styles.container}>
            <span className={styles.label}>Export:</span>
            <div className={styles.group}>
                <button className={styles.btn} onClick={handleExportCSV} title="Download CSV">
                    <FileText size={16} />
                    <span className={styles.btnText}>CSV</span>
                </button>
                <button className={styles.btn} onClick={handleExportXLSX} title="Download Excel">
                    <FileSpreadsheet size={16} />
                    <span className={styles.btnText}>Excel</span>
                </button>
                <button className={styles.btn} onClick={handleExportJSON} title="Download JSON">
                    <FileJson size={16} />
                    <span className={styles.btnText}>JSON</span>
                </button>
            </div>
        </div>
    );
};
