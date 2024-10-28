const webpack = require("webpack");

module.exports = function override(config) {
  config.resolve.fallback = {
    fs: false,
    path: require.resolve("path-browserify"),
    assert: require.resolve("assert"),
    process: require.resolve("process/browser"),
    stream: require.resolve("stream-browserify"),
    constants: require.resolve("constants-browserify"),
    util: require.resolve("util/"),
  };

  config.plugins.push(
    new webpack.ProvidePlugin({
      process: "process/browser",
      Buffer: ["buffer", "Buffer"],
    })
  );

  return config;
};
