package com

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import com.ar.ARActivity
import com.R
import com.ml.BenchmarkActivity

class MainActivity : AppCompatActivity() {
    lateinit var benchmarkButton: Button;
    lateinit var arButton: Button;
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        benchmarkButton = findViewById(R.id.benchmark)
        arButton = findViewById<Button>(R.id.arscene)
        arButton.setOnClickListener(View.OnClickListener { v: View? ->startActivity(Intent(this@MainActivity,ARActivity::class.java))})
        benchmarkButton.setOnClickListener(View.OnClickListener { v: View? ->startActivity(Intent(this@MainActivity,BenchmarkActivity::class.java))})
    }
}