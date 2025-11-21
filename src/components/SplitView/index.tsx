import React, { useState, useRef, useEffect } from 'react';
import styles from './SplitView.module.css';

interface SplitViewProps {
    leftPane: React.ReactNode;
    rightPane: React.ReactNode;
    initialLeftWidth?: number; // Percentage
}

export const SplitView: React.FC<SplitViewProps> = ({
    leftPane,
    rightPane,
    initialLeftWidth = 40
}) => {
    const [leftWidth, setLeftWidth] = useState(initialLeftWidth);
    const [isDragging, setIsDragging] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    const handleMouseDown = (e: React.MouseEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (!isDragging || !containerRef.current) return;

            const containerRect = containerRef.current.getBoundingClientRect();
            const newLeftWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100;

            // Limit width between 20% and 80%
            if (newLeftWidth >= 20 && newLeftWidth <= 80) {
                setLeftWidth(newLeftWidth);
            }
        };

        const handleMouseUp = () => {
            setIsDragging(false);
        };

        if (isDragging) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
        } else {
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }

        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isDragging]);

    return (
        <div className={styles.container} ref={containerRef}>
            <div className={styles.leftPane} style={{ width: `${leftWidth}%` }}>
                {leftPane}
            </div>

            <div
                className={`${styles.resizer} ${isDragging ? styles.active : ''}`}
                onMouseDown={handleMouseDown}
            >
                <div className={styles.handle} />
            </div>

            <div className={styles.rightPane} style={{ width: `${100 - leftWidth}%` }}>
                {rightPane}
            </div>
        </div>
    );
};
