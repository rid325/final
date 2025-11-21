import React, { useState, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from 'lucide-react';
import type { BoundingBox } from '../../lib/types';
import styles from './PDFViewer.module.css';

// Set worker source
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface PDFViewerProps {
    file: File | string;
    boundingBoxes: BoundingBox[];
    selectedRegionId?: string;
    onRegionClick: (regionId: string) => void;
}

export const PDFViewer: React.FC<PDFViewerProps> = ({
    file,
    boundingBoxes,
    selectedRegionId,
    onRegionClick
}) => {
    const [numPages, setNumPages] = useState<number>(0);
    const [pageNumber, setPageNumber] = useState<number>(1);
    const [scale, setScale] = useState<number>(1.0);
    const containerRef = useRef<HTMLDivElement>(null);
    const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);

    const isImage = file instanceof File && file.type.startsWith('image/');
    const imageUrl = isImage ? URL.createObjectURL(file) : null;

    const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
        setNumPages(numPages);
    };

    const onImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
        const { naturalWidth, naturalHeight } = e.currentTarget;
        setImageDimensions({ width: naturalWidth, height: naturalHeight });
    };

    const handleZoomIn = () => setScale(prev => Math.min(prev + 0.1, 3.0));
    const handleZoomOut = () => setScale(prev => Math.max(prev - 0.1, 0.5));

    // Render bounding boxes overlay
    const renderOverlay = () => {
        const boxesToRender = isImage
            ? boundingBoxes
            : boundingBoxes.filter(box => box.page === pageNumber);

        return boxesToRender.map(box => (
            <div
                key={box.id}
                className={`${styles.boundingBox} ${styles[box.type]} ${selectedRegionId === box.id ? styles.selected : ''}`}
                style={{
                    left: `${box.coordinates.x}%`,
                    top: `${box.coordinates.y}%`,
                    width: `${box.coordinates.width}%`,
                    height: `${box.coordinates.height}%`
                }}
                onClick={(e) => {
                    e.stopPropagation();
                    onRegionClick(box.id);
                }}
                title={box.label || box.type}
            />
        ));
    };

    return (
        <div className={styles.container} ref={containerRef}>
            <div className={styles.toolbar}>
                {!isImage && (
                    <div className={styles.pagination}>
                        <button
                            className={styles.toolBtn}
                            disabled={pageNumber <= 1}
                            onClick={() => setPageNumber(prev => prev - 1)}
                        >
                            <ChevronLeft size={20} />
                        </button>
                        <span className={styles.pageInfo}>
                            Page {pageNumber} of {numPages || '--'}
                        </span>
                        <button
                            className={styles.toolBtn}
                            disabled={pageNumber >= numPages}
                            onClick={() => setPageNumber(prev => prev + 1)}
                        >
                            <ChevronRight size={20} />
                        </button>
                    </div>
                )}
                {isImage && <div className={styles.pagination}><span className={styles.pageInfo}>Image View</span></div>}

                <div className={styles.zoomControls}>
                    <button className={styles.toolBtn} onClick={handleZoomOut}>
                        <ZoomOut size={20} />
                    </button>
                    <span className={styles.zoomLevel}>{Math.round(scale * 100)}%</span>
                    <button className={styles.toolBtn} onClick={handleZoomIn}>
                        <ZoomIn size={20} />
                    </button>
                </div>
            </div>

            <div className={styles.documentWrapper}>
                {isImage && imageUrl ? (
                    <div
                        className={styles.pageContainer}
                        style={{
                            width: imageDimensions ? imageDimensions.width * scale : 'auto',
                            height: imageDimensions ? imageDimensions.height * scale : 'auto',
                            position: 'relative',
                            margin: '0 auto'
                        }}
                    >
                        <img
                            src={imageUrl}
                            alt="Uploaded content"
                            onLoad={onImageLoad}
                            style={{
                                width: '100%',
                                height: '100%',
                                display: 'block'
                            }}
                        />
                        <div className={styles.overlay}>
                            {renderOverlay()}
                        </div>
                    </div>
                ) : (
                    <Document
                        file={file}
                        onLoadSuccess={onDocumentLoadSuccess}
                        loading={<div className={styles.loading}>Loading PDF...</div>}
                        className={styles.document}
                        onLoadError={(error) => console.error('PDF Load Error:', error)}
                    >
                        <div className={styles.pageContainer}>
                            <Page
                                pageNumber={pageNumber}
                                scale={scale}
                                renderTextLayer={false}
                                renderAnnotationLayer={false}
                                className={styles.page}
                            />
                            <div className={styles.overlay}>
                                {renderOverlay()}
                            </div>
                        </div>
                    </Document>
                )}
            </div>
        </div>
    );
};
