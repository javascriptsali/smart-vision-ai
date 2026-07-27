"""
Analytics Module for detection visualization and reporting.
Generates charts, metrics, and detailed reports from detection results.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List


def create_class_distribution_chart(class_counts: Dict[str, int]) -> go.Figure:
    """
    Create a bar chart showing the distribution of detected object classes.
    
    Args:
        class_counts: Dictionary mapping class names to their counts
        
    Returns:
        Plotly Figure object
    """
    if not class_counts:
        fig = go.Figure()
        fig.add_annotation(
            text="No objects detected", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5, 
            showarrow=False, 
            font=dict(size=20)
        )
        return fig
    
    # Convert to DataFrame and sort by count
    df = pd.DataFrame([
        {'Class': k, 'Count': v}
        for k, v in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    ])
    
    # Create bar chart
    fig = px.bar(
        df, 
        x='Class', 
        y='Count',
        title='Detected Objects Distribution',
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        margin=dict(l=20, r=20, t=50, b=100)
    )
    
    return fig


def create_confidence_histogram(detections: List[Dict]) -> go.Figure:
    """
    Create a histogram of detection confidence scores.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Plotly Figure object
    """
    if not detections:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5, 
            showarrow=False
        )
        return fig
    
    # Extract confidence scores
    confidences = [d['confidence'] for d in detections]
    
    # Create histogram
    fig = px.histogram(
        x=confidences,
        nbins=20,
        title='Confidence Score Distribution',
        labels={'x': 'Confidence', 'y': 'Count'},
        color_discrete_sequence=['#636EFA']
    )
    
    fig.update_layout(height=350)
    
    # Add threshold line
    fig.add_vline(
        x=0.5, 
        line_dash="dash", 
        line_color="orange",
        annotation_text="Default Threshold"
    )
    
    return fig


def create_summary_metrics(analytics: Dict, image_size: tuple = None) -> Dict:
    """
    Generate summary metric cards for display.
    
    Args:
        analytics: Analytics dictionary from detector
        image_size: Optional tuple of (width, height)
        
    Returns:
        Dictionary of metric labels and values
    """
    metrics = {
        'Total Objects': analytics.get('unique_classes', 0),
        'Unique Classes': analytics.get('unique_classes', 0),
        'Avg Confidence': f"{analytics.get('avg_confidence', 0) * 100:.1f}%"
    }
    
    if analytics.get('most_common'):
        cls, count = analytics['most_common']
        metrics['Most Common'] = f"{cls} ({count})"
    
    if image_size:
        metrics['Image Size'] = f"{image_size[0]}x{image_size[1]}"
    
    return metrics


def generate_detection_report(detections: List[Dict]) -> pd.DataFrame:
    """
    Generate a detailed tabular report of all detections.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Pandas DataFrame with detection details
    """
    if not detections:
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(detections)
    df = df[['class', 'confidence', 'bbox']]
    df.columns = ['Class', 'Confidence', 'Bounding Box']
    
    # Format confidence as percentage
    df['Confidence'] = df['Confidence'].apply(lambda x: f"{x * 100:.1f}%")
    
    # Format bounding box
    df['Bounding Box'] = df['Bounding Box'].apply(
        lambda x: f"({int(x[0])}, {int(x[1])}) - ({int(x[2])}, {int(x[3])})"
    )
    
    return df


# Quick test function
def test_analytics():
    """Test the analytics module with sample data."""
    print("\n=== Testing Analytics Module ===\n")
    
    # Sample detection data
    sample_detections = [
        {'class': 'person', 'confidence': 0.95, 'bbox': [100, 100, 200, 300]},
        {'class': 'person', 'confidence': 0.87, 'bbox': [300, 150, 400, 350]},
        {'class': 'car', 'confidence': 0.92, 'bbox': [500, 200, 700, 400]},
        {'class': 'dog', 'confidence': 0.78, 'bbox': [150, 400, 250, 500]},
    ]
    
    # Generate analytics
    class_counts = {}
    confidences = []
    for det in sample_detections:
        cls = det['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
        confidences.append(det['confidence'])
    
    analytics = {
        'class_counts': class_counts,
        'avg_confidence': sum(confidences) / len(confidences),
        'unique_classes': len(class_counts),
        'most_common': max(class_counts.items(), key=lambda x: x[1])
    }
    
    # Test summary metrics
    print("📊 Summary Metrics:")
    metrics = create_summary_metrics(analytics, (640, 480))
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    # Test charts (just verify they can be created)
    print("\n📈 Creating charts...")
    fig1 = create_class_distribution_chart(class_counts)
    print(f"   ✓ Class distribution chart created")
    
    fig2 = create_confidence_histogram(sample_detections)
    print(f"   ✓ Confidence histogram created")
    
    # Test report
    print("\n📋 Detection Report:")
    df = generate_detection_report(sample_detections)
    print(df.to_string(index=False))
    
    print("\n✅ Analytics test completed successfully!")


if __name__ == "__main__":
    test_analytics()