import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';
import styles from './UploadZone.module.css';

interface UploadZoneProps {
    onUploadComplete: (file: File) => void;
}

export const UploadZone: React.FC<UploadZoneProps> = ({ onUploadComplete }) => {
    const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'error'>('idle');
    const [progress, setProgress] = useState(0);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    const onDrop = useCallback((acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (!file) return;

        if (file.type !== 'application/pdf' && !file.type.startsWith('image/')) {
            setErrorMessage('Please upload a PDF or image file.');
            setUploadStatus('error');
            return;
        }

        setErrorMessage(null);
        setUploadStatus('uploading');
        simulateUploadProcess(file);
    }, [onUploadComplete]);

    const simulateUploadProcess = (file: File) => {
        // Simulate upload progress
        let currentProgress = 0;
        const interval = setInterval(() => {
            currentProgress += 10;
            setProgress(currentProgress);

            if (currentProgress >= 100) {
                clearInterval(interval);
                setUploadStatus('processing');

                // Simulate processing delay (OCR, Extraction)
                setTimeout(() => {
                    onUploadComplete(file);
                }, 1500);
            }
        }, 200);
    };

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        multiple: false,
        accept: {
            'application/pdf': ['.pdf'],
            'image/*': ['.png', '.jpg', '.jpeg']
        }
    });

    return (
        <div className={styles.container}>
            <div
                {...getRootProps()}
                className={`${styles.dropzone} ${isDragActive ? styles.active : ''} ${uploadStatus === 'error' ? styles.error : ''}`}
            >
                <input {...getInputProps()} />

                <div className={styles.content}>
                    {uploadStatus === 'idle' || uploadStatus === 'error' ? (
                        <>
                            <div className={styles.iconWrapper}>
                                <Upload className={styles.icon} size={48} />
                            </div>
                            <h3 className={styles.title}>Upload your report</h3>
                            <p className={styles.subtitle}>
                                Drag & drop a PDF or image here, or click to select
                            </p>
                            <p className={styles.hint}>Supports PDF, PNG, JPG</p>
                        </>
                    ) : (
                        <div className={styles.statusWrapper}>
                            {uploadStatus === 'uploading' && (
                                <>
                                    <Loader2 className={`${styles.icon} ${styles.spin}`} size={48} />
                                    <h3 className={styles.title}>Uploading... {progress}%</h3>
                                    <div className={styles.progressBar}>
                                        <div className={styles.progressFill} style={{ width: `${progress}%` }} />
                                    </div>
                                </>
                            )}
                            {uploadStatus === 'processing' && (
                                <>
                                    <CheckCircle2 className={`${styles.icon} ${styles.success}`} size={48} />
                                    <h3 className={styles.title}>Processing Document...</h3>
                                    <p className={styles.subtitle}>Extracting tables and charts</p>
                                </>
                            )}
                        </div>
                    )}

                    {errorMessage && (
                        <div className={styles.errorMsg}>
                            <AlertCircle size={16} />
                            <span>{errorMessage}</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
