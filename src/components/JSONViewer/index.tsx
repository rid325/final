import React from 'react';
import ReactJson from 'react-json-view';
import type { ExtractedData } from '../../lib/appTypes';
import styles from './JSONViewer.module.css';

interface JSONViewerProps {
    data: ExtractedData;
}

export const JSONViewer: React.FC<JSONViewerProps> = ({ data }) => {
    // Transform data to a cleaner JSON structure for display
    const jsonDisplay = {
        metadata: data.metadata,
        data: data.rows,
        errors: data.errors
    };

    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <h3 className={styles.title}>Raw Data View</h3>
            </div>
            <div className={styles.content}>
                <ReactJson
                    src={jsonDisplay}
                    theme="rjv-default"
                    iconStyle="triangle"
                    enableClipboard={true}
                    displayDataTypes={false}
                    style={{ backgroundColor: 'transparent' }}
                />
            </div>
        </div>
    );
};
