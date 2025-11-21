import React from 'react';
import { Sparkles, TrendingUp, AlertCircle, FileText } from 'lucide-react';
import type { SummaryInsight } from '../../lib/appTypes';
import styles from './SummaryWidget.module.css';

interface SummaryWidgetProps {
    insights: SummaryInsight[];
    isLoading?: boolean;
}

export const SummaryWidget: React.FC<SummaryWidgetProps> = ({ insights, isLoading = false }) => {
    const getIcon = (type: SummaryInsight['type']) => {
        switch (type) {
            case 'trend': return <TrendingUp size={18} className={styles.iconTrend} />;
            case 'anomaly': return <AlertCircle size={18} className={styles.iconAnomaly} />;
            default: return <FileText size={18} className={styles.iconSummary} />;
        }
    };

    if (isLoading) {
        return (
            <div className={styles.container}>
                <div className={styles.header}>
                    <Sparkles className={styles.headerIcon} size={20} />
                    <h3 className={styles.title}>AI Analysis</h3>
                </div>
                <div className={styles.loading}>
                    <div className={styles.skeleton} />
                    <div className={styles.skeleton} />
                    <div className={styles.skeleton} />
                </div>
            </div>
        );
    }

    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <Sparkles className={styles.headerIcon} size={20} />
                <h3 className={styles.title}>AI Analysis</h3>
            </div>

            <div className={styles.list}>
                {insights.map(insight => (
                    <div key={insight.id} className={styles.card}>
                        <div className={styles.cardHeader}>
                            {getIcon(insight.type)}
                            <span className={styles.cardType}>{insight.type.toUpperCase()}</span>
                        </div>
                        <p className={styles.cardContent}>{insight.content}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};
